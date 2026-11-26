"""
Guardrail：規則 → 4 個 OvR transformer → 語意比對。

import 本模組不會載入 torch / transformers / sentence-transformers，也不會改 torch 的執行緒數；
這些都延遲到第一次分類（或 main.py 啟動預熱）時才做。好處：
- 只用到規則（rule_first）或只想跑測試時，不需要安裝或載入深度學習套件
- 啟動與測試不會因為 import 就吃掉數秒和數百 MB 記憶體
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse
from collections import defaultdict
import json
import logging
import re
import os

import numpy as np

if TYPE_CHECKING:  # 只供型別標註，執行時不 import
    import torch
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =============================
# Device config (default CPU) -- change with env GUARDRAIL_DEVICE=cuda
# =============================
@lru_cache(maxsize=1)
def get_device() -> "torch.device":
    """第一次呼叫時才 import torch、決定裝置並設定 CPU 執行緒數（GUARDRAIL_DEVICE / GUARDRAIL_CPU_THREADS）。"""
    import torch

    requested = os.environ.get("GUARDRAIL_DEVICE", "cpu").lower()
    if requested in ("cuda", "gpu") and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("GUARDRAIL_CPU_THREADS", 4)))
    logger.info("guardrail: using DEVICE=%s, cpu_threads=%s", device, torch.get_num_threads())
    return device

# =============================
# Labels / Paths / Types
# =============================
LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]

def _ensure_path(p):
    return p if isinstance(p, Path) else Path(p)

# If you exported a custom model dir via env, prefer it
_model_dir_env = os.environ.get("GUARDRAIL_MODEL_DIR", None)

if _model_dir_env:
    MODEL_DIR = Path(_model_dir_env)
else:
    # Use the path you gave: classification/transformer/backend_ml_ovr_transformer_models_clean
    MODEL_DIR = Path(__file__).resolve().parents[2] / "classification" / "transformer" / "backend_ml_ovr_transformer_models_clean"

MODEL_DIR = _ensure_path(MODEL_DIR)
# 相對於本檔案定位，避免受工作目錄（CWD）影響——不同目錄啟動會 silently 跳過語意階段
semantic_path = Path(__file__).resolve().parents[2] / "semantic_models"

# CONFIG_PATH should point to the ovr_config.json file inside MODEL_DIR.
_config_env = os.environ.get("GUARDRAIL_CONFIG_PATH", None)
if _config_env:
    CONFIG_PATH = Path(_config_env)
else:
    CONFIG_PATH = MODEL_DIR / "ovr_config.json"

CONFIG_PATH = _ensure_path(CONFIG_PATH)

# Semantic artifact paths (normalized)
SEMANTIC_TEXTS_PATH = _ensure_path(semantic_path / "semantic_texts.joblib")
SEMANTIC_LABELS_PATH = _ensure_path(semantic_path / "semantic_labels.joblib")
SEMANTIC_EMB_PATH = _ensure_path(semantic_path / "semantic_embeddings.npy")



def _log_resource_paths() -> None:
    """載入資源前記錄路徑與是否存在（啟動排查用；不在 import 時做）。"""
    for name, p in (
        ("MODEL_DIR", MODEL_DIR),
        ("CONFIG_PATH", CONFIG_PATH),
        ("SEMANTIC_TEXTS_PATH", SEMANTIC_TEXTS_PATH),
        ("SEMANTIC_LABELS_PATH", SEMANTIC_LABELS_PATH),
        ("SEMANTIC_EMB_PATH", SEMANTIC_EMB_PATH),
    ):
        logger.info("guardrail: %s=%r (exists=%s)", name, p, p.exists())

ModelMap = Dict[str, Any]
ThresholdMap = Dict[str, float]

# =============================
# Whitelist & Rules
# =============================
WHITELIST_DOMAINS = {
    "samsung.com",
}

PROMPT_ATTACK_RULES = [
    # 例：ignore all instructions / ignore previous instructions / ignore all previous instructions
    r"ignore\s+(all\s+|previous\s+|your\s+|these\s+)*instructions",
    r"reveal (your|the) system prompt",
    r"system prompt",
    r"developer message",
    r"無視前面的規則",
    r"系統提示詞",
    r"繞過限制",
]

ABUSIVE_RULES = [
    r"白癡", r"智障", r"幹你娘", r"廢物", r"低能", r"講三小",
    # 「垃圾」只在罵人語境才算（垃圾分類、一般垃圾、垃圾郵件等正常詢問不能擋）：
    # 前面接「是/真/超/很/這個…」，或後面接對象（客服/公司/服務/東西…），或單獨一句「垃圾！」
    r"(是|真|超|很|有夠|根本|都|像|這個|這種|什麼|一群|一堆)垃圾(?!分類|桶|袋|車|場|郵件|信|訊息|簡訊|電話|處理|回收|清運|費|食物)"
    r"|垃圾(客服|公司|服務|東西|產品|系統|店家|商家|AI|機器人|App|app|網站|爛)"
    r"|^\s*垃圾\s*[!！。.~～]*\s*$",
]

# 強規則：單獨出現即判 SPAM。只放「正常客服對話幾乎不會出現」的片語；
# 單一常用詞（立即、投資、賺錢、入會、禮品卡…）一律不放這裡，否則會誤擋正常提問
# （例：「請立即幫我處理退款」「想了解投資型保單」「我要加入會員」）。
SPAM_RULES_STRONG = [
    # 色情 / 成人引流
    r"看片.*(找我|私訊|加我)",
    r"(高清|無碼|外流).{0,6}(影片|資源|視頻)",
    r"(AV片|偷拍自拍|情色影片|成人影片)",
    r"(私訊|加我).{0,10}(line|wechat|tg|telegram|whatsapp)",
    # 短網址服務（常見於釣魚）
    r"(?<![\w.])(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly)/",
    # 引流加群
    r"(加入我們的?群|加入群組|掃描二維碼|掃碼加|加群|入群)",
    # 明確的獲利保證話術
    r"(保證獲利|穩賺不賠|穩賺|躺賺|快速致富|本金翻倍|日入\s*\d|月入\s*\d)",
    # 灌水 / 代儲代充 / 博弈教學
    r"(買粉|買讚|刷評價|代儲|代充|下注教學|博弈網站)",
]

# 意圖規則：只有在「同時帶非白名單網址」時才判 SPAM（見 spam_rule_with_url_context）
SPAM_INTENT_PATTERNS = [
    # 聯絡/私訊/加群/加line/加tg
    r"加我(好友|line|wechat|tg|telegram|whatsapp)?",
    r"私訊我",
    r"私聊我",
    r"加入",
    r"(掃描|掃碼|掃描二維碼|掃描QR|掃描QR碼)",
    # 投資/賺錢相關
    r"保證獲利|躺賺|快速致富|快速賺錢|賺錢方法|投資賺錢|投資顧問|高報酬|高回報|零風險|無風險|年化",
    r"投資|賺錢|日入|月入|年入",
    # 促使點擊鏈接/立即行動
    r"(點我|點此|點擊|點開|按下方|按我|立刻|立即|馬上領|現在就|click here|follow the link)",
    # 購買/下單/刷單/兼職/代做
    r"(兼職|日薪|日結|接單|接案|刷單|代刷|代做)",
    # 博弈
    r"(博弈|賭博|投注|下注)",
    # 禮品卡 / 兌換碼
    r"(禮品卡|gift card|voucher|redeem code|giftcode)",
    # 加密貨幣/空投/投資工具誘導
    r"(crypto|bitcoin|btc|ethereum|eth|airdrop|空投|挖礦|加密貨幣)",
    # 詐騙常用英文片語
    r"\b(urgent|limited time|act now|verify your account|confirm your account|account suspended|claim your prize)\b",
    # 常見誘導詞（中文）
    r"(保本|無需投入|只要分享|下單即可|傳給朋友|分享即可獲得)",
]

URL_REGEX = re.compile(r"https?://[^\s]+", re.IGNORECASE)

# =============================
# Runtime state (for /health visibility)
# =============================
# 模型資源載入狀態：載入失敗時 /health 需能顯示 degraded，而非 silent fail-open。
_GUARDRAIL_STATE = {"loaded": False, "error": None, "semantic": None, "semantic_error": None}

# 載入失敗後的冷卻時間：lru_cache 不快取例外，沒有冷卻的話模型壞掉時「每一則訊息」都會重新載入一次
# （可能是好幾秒的磁碟 I/O），冷卻期間直接走降級（規則）流程。
LOAD_RETRY_COOLDOWN_SECONDS = float(os.environ.get("GUARDRAIL_LOAD_RETRY_SECONDS", "30"))
_LOAD_FAILURE: Dict[str, Any] = {"at": None, "error": None}


def guardrail_status() -> dict:
    """回報 guardrail 資源載入狀態（供 /health 使用，不觸發載入）。"""
    return {
        "loaded": bool(_GUARDRAIL_STATE["loaded"]),
        "error": _GUARDRAIL_STATE["error"],
        # 語意比對（第三層）：None = 還沒載入過；False = 模型已載入但語意階段缺檔或下載失敗而跳過
        "semantic": _GUARDRAIL_STATE["semantic"],
        "semantic_error": _GUARDRAIL_STATE["semantic_error"],
    }


_LFS_POINTER_PREFIX = b"version https://git-lfs"


def _is_lfs_pointer(path: Path) -> bool:
    """clone 時沒有 Git LFS，權重檔只會是幾百 bytes 的指標檔。"""
    try:
        if not path.is_file() or path.stat().st_size > 1024:
            return False
        with path.open("rb") as f:
            return f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _check_not_lfs_pointers(paths) -> None:
    pointers = [str(p) for p in paths if _is_lfs_pointer(Path(p))]
    if pointers:
        raise FileNotFoundError(
            "guardrail 檔案是 Git LFS 指標檔、不是真的模型（clone 時沒有 LFS）。"
            "請在倉庫根目錄執行 `git lfs install && git lfs pull`。檔案：" + ", ".join(pointers)
        )


def _fail_closed_enabled() -> bool:
    # GUARDRAIL_FAIL_CLOSED=true 時，模型不可用且規則未命中 → 以 PROMPT_ATTACK 拒答（fail-closed）
    return os.environ.get("GUARDRAIL_FAIL_CLOSED", "").strip().lower() in ("1", "true", "yes", "on")


# =============================
# Utility
# =============================
def _semantic_thresholds() -> Dict[str, float]:
    return {
        "SPAM": 0.64,
        "ABUSIVE": 0.70,
        "PROMPT_ATTACK": 0.72,
        "NORMAL": 0.68,
    }


def extract_urls(text: str) -> List[str]:
    return URL_REGEX.findall(text or "")


def _normalize_domain(host: str) -> str:
    """Normalize host: strip userinfo/port, leading www., trailing dot, lower-case."""
    if not host:
        return ""
    host = host.rsplit("@", 1)[-1].split(":")[0].strip().rstrip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_whitelisted(domain: str, whitelist_domains: set[str]) -> bool:
    """白名單比對：完全相同或為其子網域（support.samsung.com 屬於 samsung.com）。"""
    if not domain:
        return False
    return any(domain == w or domain.endswith("." + w) for w in whitelist_domains)


def _url_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return _normalize_domain(parsed.netloc or parsed.path)
    except Exception:
        return ""


def is_whitelisted_url(url: str, whitelist_domains: set[str]) -> bool:
    return _domain_whitelisted(_url_domain(url), whitelist_domains)


def _block_any_url_enabled() -> bool:
    # GUARDRAIL_BLOCK_ANY_URL=true：恢復舊行為（任何非白名單網址一律判 SPAM）
    return os.environ.get("GUARDRAIL_BLOCK_ANY_URL", "").strip().lower() in ("1", "true", "yes", "on")


def default_whitelist_domains() -> set[str]:
    """內建白名單 + GUARDRAIL_WHITELIST_DOMAINS（逗號分隔）"""
    extra = os.environ.get("GUARDRAIL_WHITELIST_DOMAINS", "")
    domains = set(WHITELIST_DOMAINS) | {d for d in (x.strip() for x in extra.split(",")) if d}
    return {_normalize_domain(d) for d in domains}


def spam_rule_with_url_context(text: str, whitelist_domains: set[str]) -> Optional[Tuple[str, float, str]]:
    t = (text or "").strip()
    urls = extract_urls(t)
    non_white_urls = [u for u in urls if not is_whitelisted_url(u, whitelist_domains)]

    for p in SPAM_RULES_STRONG:
        if re.search(p, t, re.IGNORECASE):
            return "SPAM", 0.92, f"rule:{p}"

    intent_hit = None
    for p in SPAM_INTENT_PATTERNS:
        if re.search(p, t, re.IGNORECASE):
            intent_hit = p
            break

    if len(non_white_urls) >= 2:
        return "SPAM", 0.90, "rule:multi_non_whitelist_urls"

    if len(non_white_urls) == 1 and intent_hit:
        return "SPAM", 0.88, f"rule:single_url_with_intent:{intent_hit}"

    return None


def _first_non_whitelisted_url(text: str, whitelist_domains: set[str]) -> Optional[Tuple[str, str]]:
    """Return (url, domain) for the first URL that is NOT whitelisted, else None."""
    for u in extract_urls(text or ""):
        domain = _url_domain(u)
        if not _domain_whitelisted(domain, whitelist_domains):
            return (u, domain)
    return None


def rule_first(text: str, whitelist_domains: set[str]) -> Optional[Tuple[str, float, str]]:
    """
    RULE stage:
    0) (僅 GUARDRAIL_BLOCK_ANY_URL=true) 任何非白名單網址 -> SPAM
    1) PROMPT_ATTACK / ABUSIVE regex checks
    2) spam_rule_with_url_context：強規則、多個非白名單網址、或網址 + 引流意圖
    預設不再因「單一網址」就直接拒答（使用者貼訂單/商品連結是正常客服情境），
    單一網址交給後續 ML / 語意階段判斷。
    """
    t = text or ""

    non_white = _first_non_whitelisted_url(t, whitelist_domains) if _block_any_url_enabled() else None
    if non_white:
        url, domain = non_white
        reason = f"rule:non_whitelist_url:{domain or 'unknown'}"
        logger.info(f"guardrail: blocking non-whitelist url detected: {url} domain={domain}")
        return "SPAM", 0.99, reason

    # 1) existing checks
    for p in PROMPT_ATTACK_RULES:
        if re.search(p, t, re.IGNORECASE):
            return "PROMPT_ATTACK", 0.97, f"rule:{p}"

    for p in ABUSIVE_RULES:
        if re.search(p, t, re.IGNORECASE):
            return "ABUSIVE", 0.93, f"rule:{p}"

    # 2) existing spam url intent checks
    spam_hit = spam_rule_with_url_context(t, whitelist_domains)
    if spam_hit:
        return spam_hit

    return None
# ---------- end override ----------


# =============================
# ML (OvR) stage - Transformer (CPU friendly)
# =============================
def _predict_ovr_transformer(
    models: ModelMap,
    tokenizers: Dict[str, Any],
    thresholds: ThresholdMap,
    text: str,
    device: "torch.device",
    max_len: int,
    shared_tokenizer: Any = None,
) -> Tuple[str, float, Dict[str, float], Dict[str, bool]]:
    """
    Run each transformer binary model and return:
      - selected label (or 'UNCERTAIN')
      - confidence (prob of selected or top)
      - probs: map label -> pos-prob
      - pass_flags: map label -> bool (pos-prob >= threshold)

    四個二元模型是不同的權重，沒辦法合併成一次 forward；但它們從同一個預訓練模型微調，
    tokenizer 相同時（shared_tokenizer）只 tokenize 一次、只搬一次裝置。
    """
    import torch

    probs: Dict[str, float] = {}
    pass_flags: Dict[str, bool] = {}

    def _encode(tok: Any) -> Dict[str, Any]:
        enc = tok(text, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    shared_enc = _encode(shared_tokenizer) if shared_tokenizer is not None else None

    with torch.inference_mode():
        for lb in LABELS:
            enc = shared_enc if shared_enc is not None else _encode(tokenizers[lb])
            logits = models[lb](**enc).logits  # shape [1, 2]
            prob_pos = float(torch.softmax(logits, dim=1)[0, 1].item())
            probs[lb] = prob_pos
            pass_flags[lb] = prob_pos >= float(thresholds.get(lb, 0.5))

    passed = [lb for lb in LABELS if pass_flags[lb]]
    if passed:
        best = max(passed, key=lambda x: probs[x])
        return best, probs[best], probs, pass_flags

    # none passed threshold -> return UNCERTAIN with highest prob
    best_any = max(LABELS, key=lambda x: probs[x])
    return "UNCERTAIN", probs[best_any], probs, pass_flags


def _same_tokenizer(tokenizers: Dict[str, Any]) -> Any:
    """四個 tokenizer 的類別與詞表都相同時回傳其中一個，否則 None（逐一 tokenize）。"""
    toks = [tokenizers[lb] for lb in LABELS]
    first = toks[0]
    try:
        vocab = first.get_vocab()
        if all(type(t) is type(first) and t.get_vocab() == vocab for t in toks[1:]):
            return first
    except Exception:  # 自訂 tokenizer 沒有 get_vocab 等情況
        pass
    return None


# =============================
# Load resources (Transformer models + semantic resources)
# =============================
@lru_cache(maxsize=1)
def _load_guardrail_resources():
    _log_resource_paths()
    # 先檢查是不是 LFS 指標檔（不需要 transformers 就能給出明確的錯誤）
    _check_not_lfs_pointers(
        [p for lb in LABELS for p in sorted((MODEL_DIR / lb.lower()).glob("*.safetensors"))
         + sorted((MODEL_DIR / lb.lower()).glob("*.bin"))]
    )

    # 重量級套件在這裡才 import（見模組說明）
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # 1) OVR config
    cfg_path = Path(CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    thresholds: ThresholdMap = cfg["thresholds"]
    max_len = int(cfg.get("max_len", 256))
    embedding_model_name = cfg.get("embedding_model_name", "paraphrase-multilingual-MiniLM-L12-v2")
    semantic_topk = int(cfg.get("semantic_topk", 5))

    # 2) Load transformer models + tokenizers per label
    device = get_device()  # CPU by default
    models: ModelMap = {}
    tokenizers: Dict[str, Any] = {}

    for lb in LABELS:
        subdir = MODEL_DIR / lb.lower()
        if not subdir.exists():
            raise FileNotFoundError(f"Model folder missing: {subdir}")
        tokenizer = AutoTokenizer.from_pretrained(str(subdir))
        # low_cpu_mem_usage reduces peak usage during load (requires newer transformers)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(subdir),
            low_cpu_mem_usage=True,
        )
        # Move model to chosen device (CPU)
        model.to(device)
        model.eval()
        tokenizers[lb] = tokenizer
        models[lb] = model
        logger.info(f"guardrail: loaded {lb} model to {device}")

    # 3) Semantic artifacts (optional)
    embedder = None
    corpus_texts: List[str] = []
    corpus_labels: List[str] = []
    corpus_emb: Optional[np.ndarray] = None
    semantic_error: Optional[str] = None
    try:
        if SEMANTIC_TEXTS_PATH.exists() and SEMANTIC_LABELS_PATH.exists() and SEMANTIC_EMB_PATH.exists():
            _check_not_lfs_pointers([SEMANTIC_TEXTS_PATH, SEMANTIC_LABELS_PATH, SEMANTIC_EMB_PATH])
            import joblib as _joblib  # local import to avoid heavy dependency if unused
            corpus_texts = _joblib.load(SEMANTIC_TEXTS_PATH)
            corpus_labels = _joblib.load(SEMANTIC_LABELS_PATH)
            corpus_emb = np.load(SEMANTIC_EMB_PATH)
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(
                embedding_model_name,
                device="cpu"
            )

            if len(corpus_texts) != len(corpus_labels):
                raise ValueError("semantic_texts and semantic_labels length mismatch")
            if len(corpus_texts) != int(corpus_emb.shape[0]):
                raise ValueError("semantic corpus size and embedding rows mismatch")
        else:
            semantic_error = "semantic files missing"
            logger.warning("Semantic resources missing, stage-2 will be skipped.")
    except Exception as e:
        # 常見原因：第一次啟動連不到 HuggingFace，下載不到 embedding 模型（之後會存在 HF 快取）
        semantic_error = f"{type(e).__name__}: {str(e)[:200]}"
        logger.warning(f"Semantic resources unavailable, stage-2 skipped: {e}")
        embedder = None
        corpus_texts = []
        corpus_labels = []
        corpus_emb = None

    whitelist = default_whitelist_domains() | {
        _normalize_domain(d) for d in cfg.get("whitelist_domains", []) if isinstance(d, str)
    }

    _GUARDRAIL_STATE["loaded"] = True
    _GUARDRAIL_STATE["error"] = None
    _GUARDRAIL_STATE["semantic"] = embedder is not None
    _GUARDRAIL_STATE["semantic_error"] = semantic_error

    return {
        "models": models,
        "tokenizers": tokenizers,
        "shared_tokenizer": _same_tokenizer(tokenizers),
        "thresholds": thresholds,
        "max_len": max_len,
        "embedder": embedder,
        "corpus_texts": corpus_texts,
        "corpus_labels": corpus_labels,
        "corpus_emb": corpus_emb,
        "semantic_topk": semantic_topk,
        "whitelist_domains": whitelist,
        "device": device,
    }


def _get_resources():
    """取得已載入的資源；最近一次載入失敗還在冷卻期內時，不重試、直接拋出同一個錯誤。"""
    failed_at = _LOAD_FAILURE["at"]
    if failed_at is not None and time.monotonic() - failed_at < LOAD_RETRY_COOLDOWN_SECONDS:
        raise RuntimeError(f"guardrail resources unavailable (retry cooldown): {_LOAD_FAILURE['error']}")
    try:
        rs = _load_guardrail_resources()
    except Exception as e:
        _LOAD_FAILURE["at"] = time.monotonic()
        _LOAD_FAILURE["error"] = str(e)
        raise
    _LOAD_FAILURE["at"] = None
    _LOAD_FAILURE["error"] = None
    return rs


def _semantic_predict(
    text: str,
    embedder: "SentenceTransformer",
    corpus_emb: np.ndarray,
    corpus_texts: List[str],
    corpus_labels: List[str],
    semantic_topk: int,
):
    q = embedder.encode([text], normalize_embeddings=True)[0]
    sims = np.dot(corpus_emb, q)

    topk_idx = np.argsort(-sims)[:semantic_topk]
    topk = [
        (int(i), float(sims[i]), corpus_labels[int(i)], corpus_texts[int(i)])
        for i in topk_idx
    ]

    scores = defaultdict(float)
    for _, sim, lb, _ in topk:
        scores[lb] += sim

    best_label = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = float(scores[best_label] / total) if total > 0 else 0.0

    return best_label, confidence, topk


# =============================
# Public API
# =============================
def classify_text(text: str) -> dict:
    """
    Three-stage classify（實際執行順序）:
    1) RULE
    2) ML OVR (transformer)
    3) SEMANTIC (if resources available; 僅在 ML 無類別過門檻時)
    模型不可用時降級為純規則；未命中規則回 UNCERTAIN（或 fail-closed 時 PROMPT_ATTACK），
    並以 available=False 標記，呼叫端不得視為 NORMAL。
    """
    try:
        rs = _get_resources()
        models = rs["models"]
        tokenizers = rs["tokenizers"]
        thresholds = rs["thresholds"]
        max_len = rs["max_len"]
        embedder = rs["embedder"]
        corpus_texts = rs["corpus_texts"]
        corpus_labels = rs["corpus_labels"]
        corpus_emb = rs["corpus_emb"]
        semantic_topk = rs["semantic_topk"]
        whitelist_domains = rs["whitelist_domains"]
        device = rs["device"]

        # Stage 1: RULE
        rr = rule_first(text, whitelist_domains)
        if rr:
            lb, cf, reason = rr
            probs = {x: (1.0 if x == lb else 0.0) for x in LABELS}
            return {
                "stage": "RULE",
                "label": lb,
                "confidence": cf,
                "reason": reason,
                "probs": probs,
                "flags": {},
                "topk": [],
                "available": True,
            }
        ml_label, ml_conf, probs, flags = _predict_ovr_transformer(
            models=models,
            tokenizers=tokenizers,
            thresholds=thresholds,
            text=text,
            device=device,
            max_len=max_len,
            shared_tokenizer=rs.get("shared_tokenizer"),
        )

        if ml_label != "UNCERTAIN":
            return {
                "stage": "ML",
                "label": ml_label,
                "confidence": ml_conf,
                "reason": f"ovr {ml_label} passed threshold",
                "probs": probs,
                "flags": flags,
                "topk": [],
                "available": True,
            }
        # Stage 3: SEMANTIC (optional)
        if embedder is not None and corpus_emb is not None and len(corpus_texts) > 0:
            sem_label, sem_conf, topk = _semantic_predict(
                text=text,
                embedder=embedder,
                corpus_emb=corpus_emb,
                corpus_texts=corpus_texts,
                corpus_labels=corpus_labels,
                semantic_topk=semantic_topk,
            )
            sem_th = _semantic_thresholds().get(sem_label, 0.70)
            if sem_conf >= sem_th:
                probs = {x: 0.0 for x in LABELS}
                probs[sem_label] = sem_conf
                return {
                    "stage": "SEMANTIC",
                    "label": sem_label,
                    "confidence": sem_conf,
                    "reason": f"semantic {sem_label} conf={sem_conf:.4f} >= th={sem_th:.2f}",
                    "probs": probs,
                    "flags": {},
                    "topk": topk,
                    "available": True,
                }

        # Final uncertain (service available, but no confident class)
        return {
            "stage": "UNCERTAIN",
            "label": "UNCERTAIN",
            "confidence": ml_conf,
            "reason": "rule/semantic not triggered and no OVR class passed threshold",
            "probs": probs,
            "flags": flags,
            "topk": [],
            "available": True,
        }

    except Exception as e:
        logger.warning(f"Guardrail classification unavailable: {e}")
        _GUARDRAIL_STATE["loaded"] = False
        _GUARDRAIL_STATE["error"] = str(e)

        # Fail-safe：不再默認放行（fail-open）。降級為「純規則」篩查（規則不依賴模型）；
        # 仍未命中時，依 GUARDRAIL_FAIL_CLOSED 決定拒答（PROMPT_ATTACK）或標記 UNCERTAIN。
        try:
            rr = rule_first(text, default_whitelist_domains())
        except Exception as rule_err:
            logger.warning(f"Guardrail rule-only stage also failed: {rule_err}")
            rr = None

        if rr:
            lb, cf, reason = rr
            logger.warning(f"Guardrail DEGRADED to rule-only; rule hit: label={lb} reason={reason}")
            probs = {x: (1.0 if x == lb else 0.0) for x in LABELS}
            return {
                "stage": "RULE_DEGRADED",
                "label": lb,
                "confidence": cf,
                "reason": f"degraded rule-only hit: {reason}",
                "probs": probs,
                "flags": {},
                "topk": [],
                "available": False,
            }

        degraded_label = "PROMPT_ATTACK" if _fail_closed_enabled() else "UNCERTAIN"
        logger.warning(
            "Guardrail DEGRADED, no rule hit; label=%s (GUARDRAIL_FAIL_CLOSED=%s)",
            degraded_label, _fail_closed_enabled(),
        )
        return {
            "stage": "DEGRADED",
            "label": degraded_label,
            "confidence": 0.0,
            "reason": f"service unavailable: {e}",
            "probs": {},
            "flags": {},
            "topk": [],
            "available": False,
        }