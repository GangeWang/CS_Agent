from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse
from collections import defaultdict
import json
import logging
import re
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =============================
# Device config (default CPU) -- change with env GUARDRAIL_DEVICE=cuda
# =============================
RequestedDevice = os.environ.get("GUARDRAIL_DEVICE", "cpu").lower()
if RequestedDevice in ("cuda", "gpu") and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

torch.set_num_threads(int(os.environ.get("GUARDRAIL_CPU_THREADS", 4)))
logger.info(f"guardrail: using DEVICE={DEVICE}, cpu_threads={torch.get_num_threads()}")

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
    # Use the path you gave: classifcation/transformer/backend_ml_ovr_transformer_models_clean
    MODEL_DIR = Path(__file__).resolve().parents[2] / "classifcation" / "transformer" / "backend_ml_ovr_transformer_models_clean"

MODEL_DIR = _ensure_path(MODEL_DIR)

# CONFIG_PATH should point to the ovr_config.json file inside MODEL_DIR.
_config_env = os.environ.get("GUARDRAIL_CONFIG_PATH", None)
if _config_env:
    CONFIG_PATH = Path(_config_env)
else:
    CONFIG_PATH = MODEL_DIR / "ovr_config.json"

CONFIG_PATH = _ensure_path(CONFIG_PATH)

# Semantic artifact paths (normalized)
SEMANTIC_TEXTS_PATH = _ensure_path(MODEL_DIR / "semantic_texts.joblib")
SEMANTIC_LABELS_PATH = _ensure_path(MODEL_DIR / "semantic_labels.joblib")
SEMANTIC_EMB_PATH = _ensure_path(MODEL_DIR / "semantic_embeddings.npy")

# Debug log right after normalization (helps startup diagnostics)
logger.info(f"guardrail: MODEL_DIR={MODEL_DIR!r} (exists={MODEL_DIR.exists()})")
logger.info(f"guardrail: CONFIG_PATH={CONFIG_PATH!r} (exists={CONFIG_PATH.exists()})")
logger.info(f"guardrail: SEMANTIC_TEXTS_PATH={SEMANTIC_TEXTS_PATH!r} (exists={SEMANTIC_TEXTS_PATH.exists()})")
logger.info(f"guardrail: SEMANTIC_LABELS_PATH={SEMANTIC_LABELS_PATH!r} (exists={SEMANTIC_LABELS_PATH.exists()})")
logger.info(f"guardrail: SEMANTIC_EMB_PATH={SEMANTIC_EMB_PATH!r} (exists={SEMANTIC_EMB_PATH.exists()})")

ModelMap = Dict[str, Any]
ThresholdMap = Dict[str, float]

# =============================
# Whitelist & Rules
# =============================
WHITELIST_DOMAINS = {
    "samsung.com",
}

PROMPT_ATTACK_RULES = [
    r"ignore (all|previous) instructions",
    r"reveal (your|the) system prompt",
    r"system prompt",
    r"developer message",
    r"無視前面的規則",
    r"系統提示詞",
    r"繞過限制",
]

ABUSIVE_RULES = [
    r"白癡", r"智障", r"垃圾", r"幹你娘", r"廢物", r"低能", r"講三小"
]

SPAM_RULES_STRONG = [
    # 原有強規則（聯絡/成人等）
    r"看片.*(找我|私訊|加我)",
    r"(高清|無碼|外流).*(影片|資源)?",
    r"(成人|AV片|絲襪|偷拍自拍|情色)",
    r"(私訊|加我).*(line|wechat|tg|telegram|whatsapp)",
    # 短連結或可疑短網址服務 (短連結常見於釣魚)
    r"\b(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|短網址|短連結)\b",
    # 直接指令性邀請加群或私下聯絡
    r"(加入我們|加入群組|掃描二維碼|掃碼|掃描QR|加群|入群|入會).*",
    # 推銷/誘導/立即行動詞常見於詐騙
    r"(點我|點此|立刻|立即|現在就|馬上領|點下方|點擊鏈接).*",
    # 提供保證獲利 / 賺錢 / 日入
    r"(保證獲利|穩賺|無風險|零風險|日入|月入|年化|躺賺|快速致富|賺大錢|本金翻倍|投資|賺錢).*",
    # 購買服務 / 虛假投資 / 博弈代儲代充
    r"(買粉|買讚|刷評價|博弈|賭博|代儲|代充|投注|下注|下注教學).*",
    # 含有看似誘導的敏感金流用語（gift card 等）
    r"(禮品卡|gift card|voucher|redeem code|giftcode).*",

]

SPAM_INTENT_PATTERNS = [
    # 聯絡/私訊/加群/加line/加tg
    r"加我(好友|line|wechat|tg|telegram|whatsapp)?",
    r"私訊我",
    r"私聊我",
    r"(加入|加入我們).*",
    r"(掃描|掃碼|掃描二維碼|掃描QR|掃描QR碼)",
    # 投資/賺錢相關
    r"保證獲利|躺賺|快速致富|快速賺錢|賺錢方法|投資賺錢|投資顧問|高報酬|高回報",
    r"日入|月入|年入|日入\W*\d+|月入\W*\d+",
    # 金融/匯款/收款/付款/轉帳/提款
    # 促使點擊鏈接/短連結/點擊操作
    r"(點我|點此|點擊|點開|按下方|按我|クリック|click here|follow the link)\b",
    r"\b(bit\.ly|tinyurl|goo\.gl|t\.co|ow\.ly)\b",
    # 購買/下單/刷單/兼職/代做
    r"(兼職|日薪|日結|接單|接案|刷單|代刷|代做).*",
    # 加密貨幣/空投/投資工具誘導
    r"(crypto|bitcoin|btc|ethereum|eth|airdrop|空投|挖礦|加密貨幣).*",
    # 詐騙常用英文片語
    r"\b(urgent|limited time|act now|verify your account|confirm your account|account suspended|claim your prize)\b",
    # 常見誘導詞（中文）
    r"(保證|零風險|保本|無需投入|只要分享|下單即可|傳給朋友|分享即可獲得).*",
]

URL_REGEX = re.compile(r"https?://[^\s]+", re.IGNORECASE)


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


def is_whitelisted_url(url: str, whitelist_domains: set[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return host in whitelist_domains
    except Exception:
        return False


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


# ---------- URL whitelist helpers & rule_first override ----------
def _normalize_domain(host: str) -> str:
    """
    Normalize host: strip port, leading www., lower-case.
    """
    if not host:
        return ""
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host.lower()

def _first_non_whitelisted_url(text: str, whitelist_domains: set[str]) -> Optional[Tuple[str, str]]:
    """
    Return (url, domain) for the first URL that is NOT in whitelist_domains.
    If none found, return None.
    """
    urls = extract_urls(text or "")
    if not urls:
        return None
    for u in urls:
        try:
            parsed = urlparse(u)
            host = parsed.netloc or parsed.path  # handle some weird urls
            domain = _normalize_domain(host)
            if domain == "":
                continue
            # whitelist_domains expected to be lower-case strings without www.
            if domain not in whitelist_domains:
                return (u, domain)
        except Exception:
            # if parsing fails, treat as non-whitelisted suspicious URL
            return (u, "")
    return None

def rule_first(text: str, whitelist_domains: set[str]) -> Optional[Tuple[str, float, str]]:
    """
    RULE stage:
    0) If any non-whitelisted URL present -> SPAM (block).
    1) PROMPT_ATTACK / ABUSIVE regex checks
    2) spam_rule_with_url_context checks (existing)
    """
    t = text or ""

    # 0) BLOCK: any non-whitelisted URL => SPAM immediately
    non_white = _first_non_whitelisted_url(t, whitelist_domains)
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
    device: torch.device,
    max_len: int,
) -> Tuple[str, float, Dict[str, float], Dict[str, bool]]:
    """
    Run each transformer binary model and return:
      - selected label (or 'UNCERTAIN')
      - confidence (prob of selected or top)
      - probs: map label -> pos-prob
      - pass_flags: map label -> bool (pos-prob >= threshold)
    """
    probs: Dict[str, float] = {}
    pass_flags: Dict[str, bool] = {}

    # single example -> tokenize per model (tokenizer may differ by model type)
    for lb in LABELS:
        tokenizer = tokenizers[lb]
        model = models[lb]

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        # Move tensors to desired device
        enc = {k: v.to(device) for k, v in enc.items()}

        # inference_mode is preferred on CPU for performance/low memory
        with torch.inference_mode():
            out = model(**enc)
            logits = out.logits  # shape [1, num_labels] (num_labels==2)
            probs_arr = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prob_pos = float(probs_arr[1])

        probs[lb] = prob_pos
        pass_flags[lb] = prob_pos >= float(thresholds.get(lb, 0.5))

    passed = [lb for lb in LABELS if pass_flags[lb]]
    if passed:
        best = max(passed, key=lambda x: probs[x])
        return best, probs[best], probs, pass_flags

    # none passed threshold -> return UNCERTAIN with highest prob
    best_any = max(LABELS, key=lambda x: probs[x])
    return "UNCERTAIN", probs[best_any], probs, pass_flags


# =============================
# Load resources (Transformer models + semantic resources)
# =============================
@lru_cache(maxsize=1)
def _load_guardrail_resources():
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
    device = DEVICE  # use global DEVICE (CPU by default)
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
    try:
        if SEMANTIC_TEXTS_PATH.exists() and SEMANTIC_LABELS_PATH.exists() and SEMANTIC_EMB_PATH.exists():
            import joblib as _joblib  # local import to avoid heavy dependency if unused
            corpus_texts = _joblib.load(SEMANTIC_TEXTS_PATH)
            corpus_labels = _joblib.load(SEMANTIC_LABELS_PATH)
            corpus_emb = np.load(SEMANTIC_EMB_PATH)
            embedder = SentenceTransformer(embedding_model_name)

            if len(corpus_texts) != len(corpus_labels):
                raise ValueError("semantic_texts and semantic_labels length mismatch")
            if len(corpus_texts) != int(corpus_emb.shape[0]):
                raise ValueError("semantic corpus size and embedding rows mismatch")
        else:
            logger.warning("Semantic resources missing, stage-2 will be skipped.")
    except Exception as e:
        logger.warning(f"Semantic resources unavailable, stage-2 skipped: {e}")
        embedder = None
        corpus_texts = []
        corpus_labels = []
        corpus_emb = None

    whitelist = set(map(str.lower, cfg.get("whitelist_domains", sorted(WHITELIST_DOMAINS))))

    return {
        "models": models,
        "tokenizers": tokenizers,
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


def _semantic_predict(
    text: str,
    embedder: SentenceTransformer,
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
    Three-stage classify:
    1) RULE
    2) SEMANTIC (if resources available)
    3) ML OVR (transformer)
    Fallback-safe NORMAL only when whole service unavailable.
    """
    try:
        rs = _load_guardrail_resources()
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

        # Stage 2: SEMANTIC (optional)
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

        # Stage 3: ML (OvR Transformer)
        ml_label, ml_conf, probs, flags = _predict_ovr_transformer(
            models=models,
            tokenizers=tokenizers,
            thresholds=thresholds,
            text=text,
            device=device,
            max_len=max_len,
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
        return {
            "stage": "FALLBACK",
            "label": "NORMAL",
            "confidence": 0.0,
            "reason": f"service unavailable: {e}",
            "probs": {},
            "flags": {},
            "topk": [],
            "available": False,
        }