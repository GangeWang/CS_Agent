# -----------------------------
# 匯入必要模組
# -----------------------------
from __future__ import annotations  # Python 3.7+ 支援前置型別註解
import argparse  # 命令列參數解析
import re  # 正則表達式
from pathlib import Path  # 路徑處理
from typing import Dict, List  # 型別提示
from urllib.parse import urlparse  # URL 解析
import joblib  # 用於讀取 sklearn / ML 模型
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # 句子嵌入模型
from collections import defaultdict  # 預設字典，用於累計分數

# -----------------------------
# 定義分類標籤
# -----------------------------
LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]

# -----------------------------
# 白名單網域（只允許這些域名的連結，不算 spam）
# -----------------------------
WHITELIST_DOMAINS = {
    "www.samsung.com"
}

# -----------------------------
# 強規則列表 (High precision, high confidence rules)
# -----------------------------
# 1) Prompt attack 關鍵詞（高危險，直接封鎖）
PROMPT_ATTACK_RULES = [
    r"ignore (all|previous) instructions",  # 英文規避指令
    r"reveal (your|the) system prompt",
    r"system prompt",
    r"developer message",
    r"無視前面的規則",  # 中文規避
    r"系統提示詞",
    r"繞過限制",
]

# 2) 辱罵詞（ABUSIVE）
ABUSIVE_RULES = [
    r"白癡", r"智障", r"垃圾", r"幹你娘", r"廢物", r"低能", r"講三小"
]

# 3) 強 spam 關鍵詞（高精度 spam）
SPAM_RULES_STRONG = [
    r"看片.*(找我|私訊|加我)",  # 色情誘導
    r"(高清|無碼|外流).*(影片|資源)?",  # 色情影片
    r"(成人|AV片|自拍偷拍)",  # 成人內容
    r"(私訊|加我).*(line|wechat|tg)",  # 社群誘導
]

# 4) spam 意圖詞（誘導 / 詐騙 / 色情）
SPAM_INTENT_PATTERNS = [
    r"加我(好友|line|wechat|tg)",
    r"私訊我",
    r"立即領獎|免費領獎|點我領",
    r"保證獲利|躺賺|日入",
    r"成人|無碼|外流|AV片|自拍偷拍",
    r"買粉|買讚|刷評價",
    r"博弈|代儲|代充",
]

# URL 正則匹配
URL_REGEX = re.compile(r"https?://[^\s]+", re.IGNORECASE)

# -----------------------------
# URL 相關工具函數
# -----------------------------
def extract_urls(text: str) -> List[str]:
    """從文字中抓出所有 URL"""
    return URL_REGEX.findall(text)

def is_whitelisted_url(url: str, whitelist_domains: set[str]) -> bool:
    """檢查 URL 是否在白名單內"""
    try:
        host = urlparse(url).netloc.lower()  # 取得 domain
        return host in whitelist_domains
    except Exception:
        return False

# -----------------------------
# 強 spam 規則 + URL 語境判斷
# -----------------------------
def spam_rule_with_url_context(text: str, whitelist_domains: set[str]):
    """
    Spam 判斷流程：
    1. 如果文字符合強 spam 關鍵詞 -> 立即封
    2. 若有意圖詞 + 非白名單網址 -> 封
    3. 多個非白名單網址 -> 封
    4. 只有白名單網址或沒有危險關鍵詞 -> 不封
    """
    t = text.strip()
    urls = extract_urls(t)
    non_white_urls = [u for u in urls if not is_whitelisted_url(u, whitelist_domains)]

    # 強 spam 關鍵句（無網址也可封）
    for p in SPAM_RULES_STRONG:
        if re.search(p, t, re.IGNORECASE):
            return "SPAM", 0.92, f"rule:{p}"

    # 意圖詞（詐騙/導流/色情）
    intent_hit = None
    for p in SPAM_INTENT_PATTERNS:
        if re.search(p, t, re.IGNORECASE):
            intent_hit = p
            break

    # 多個非白名單網址 -> 封
    if len(non_white_urls) >= 2:
        return "SPAM", 0.90, "rule:multi_non_whitelist_urls"

    # 單個非白名單網址 + 意圖詞 -> 封
    if len(non_white_urls) == 1 and intent_hit:
        return "SPAM", 0.88, f"rule:single_url_with_intent:{intent_hit}"

    # 只有一個網址（尤其白名單）不封
    return None

# -----------------------------
# 規則優先判斷 (Rule-first)
# -----------------------------
def rule_first(text: str, whitelist_domains: set[str]):
    """依序判斷 PROMPT_ATTACK、ABUSIVE、SPAM 強規則"""
    for p in PROMPT_ATTACK_RULES:
        if re.search(p, text, re.IGNORECASE):
            return "PROMPT_ATTACK", 0.97, f"rule:{p}"
    for p in ABUSIVE_RULES:
        if re.search(p, text, re.IGNORECASE):
            return "ABUSIVE", 0.93, f"rule:{p}"
    spam_hit = spam_rule_with_url_context(text, whitelist_domains)
    if spam_hit:
        return spam_hit
    return None

# -----------------------------
# 噪音文字判斷
# -----------------------------
def is_noise_text(t: str) -> bool:
    """判斷文字是否為噪音（太短、重複字元、純符號）"""
    t = t.strip()
    if len(t) <= 1:
        return True
    if re.fullmatch(r"(.)\1{5,}", t):  # 連續同字元 6 次以上
        return True
    if re.fullmatch(r"[\W_]+", t):  # 只含非文字字元
        return True
    return False

# -----------------------------
# RSMTester 類別 (Rule + Semantic + ML)
# -----------------------------
class RSMTester:
    """
    RSMTester: 三層 Guardrail 模型
    1. 強規則判斷 (Rule)
    2. 語意相似度判斷 (Semantic Top-K)
    3. ML 模型 fallback
    """
    def __init__(
        self,
        ml_model_path: Path,
        data_path: Path,
        whitelist_domains: set[str],
        embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        semantic_topk: int = 5,
    ):
        # 檢查 ML 模型與資料檔案存在
        if not ml_model_path.exists():
            raise FileNotFoundError(f"ML model not found: {ml_model_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.whitelist_domains = whitelist_domains
        self.ml_model = joblib.load(ml_model_path)  # 載入 sklearn ML 模型

        # 讀取資料檔案，清理空值 / 噪音文字 / 非標籤資料
        df = pd.read_csv(data_path).dropna(subset=["text", "label"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()
        df = df[df["text"] != ""]
        df = df[~df["text"].apply(is_noise_text)]
        df = df[df["label"].isin(LABELS)]
        df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

        # 保存語料與標籤
        self.corpus_texts = df["text"].tolist()
        self.corpus_labels = df["label"].tolist()

        # 句向量模型
        self.embedder = SentenceTransformer(embedding_model_name)
        self.corpus_emb = self.embedder.encode(self.corpus_texts, normalize_embeddings=True)
        self.semantic_topk = semantic_topk

        print(f"[init] semantic corpus rows: {len(self.corpus_texts)}")
        print(f"[init] whitelist domains   : {sorted(self.whitelist_domains)}")

    # -----------------------------
    # 語意閾值
    # -----------------------------
    def semantic_thresholds(self) -> Dict[str, float]:
        return {
            "SPAM": 0.64,
            "ABUSIVE": 0.70,
            "PROMPT_ATTACK": 0.72,
            "NORMAL": 0.68,
        }

    # -----------------------------
    # ML 閾值
    # -----------------------------
    def ml_thresholds(self) -> Dict[str, float]:
        return {
            "SPAM": 0.70,
            "ABUSIVE": 0.58,
            "PROMPT_ATTACK": 0.62,
            "NORMAL": 0.55,
        }

    # -----------------------------
    # ML 模型預測
    # -----------------------------
    def ml_predict(self, text: str):
        """使用 ML 模型進行預測"""
        probs = self.ml_model.predict_proba([text])[0]
        classes = list(self.ml_model.classes_)
        prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
        for lb in LABELS:
            prob_map.setdefault(lb, 0.0)
        best_idx = int(np.argmax(probs))
        best_label = str(classes[best_idx])
        best_conf = float(probs[best_idx])
        return best_label, best_conf, prob_map

    # -----------------------------
    # 語意相似度 Top-K 預測
    # -----------------------------
    def semantic_predict(self, text: str):
        """Semantic Top-K 加權投票 + similarity gap 檢查"""
        q = self.embedder.encode([text], normalize_embeddings=True)[0]  # 問句向量
        sims = np.dot(self.corpus_emb, q)  # 計算語意相似度

        # Top-K 最相似的文本
        topk_idx = np.argsort(-sims)[: self.semantic_topk]
        topk = [
            (int(i), float(sims[i]), self.corpus_labels[int(i)], self.corpus_texts[int(i)])
            for i in topk_idx
        ]

        # Top-K 加權投票
        scores = defaultdict(float)
        for _, sim, label, _ in topk:
            scores[label] += sim
        best_label = max(scores, key=scores.get)
        best_sim = scores[best_label]

        # similarity gap 檢查：Top1 與 Top2 相差太小 -> 不確定
        top1_sim = topk[0][1]
        top2_sim = topk[1][1] if len(topk) > 1 else 0.0
        if abs(top1_sim - top2_sim) < 0.02:
            return "UNCERTAIN", best_sim, topk[0][3], topk

        return best_label, best_sim, topk[0][3], topk

    # -----------------------------
    # 綜合分類函數 (Rule -> Semantic -> ML -> Fallback)
    # -----------------------------
    def classify(self, text: str):
        # 1. Rule 判斷
        rr = rule_first(text, self.whitelist_domains)
        if rr:
            lb, cf, rs = rr
            prob_map = {x: (1.0 if x == lb else 0.0) for x in LABELS}
            return {
                "stage": "RULE",
                "label": lb,
                "confidence": cf,
                "reason": rs,
                "probabilities": prob_map,
                "topk": [],
            }

        # 2. Semantic
        sem_label, sem_sim, sem_text, topk = self.semantic_predict(text)

        # Top-K 加權分數
        scores = {}
        for idx, sim, lb, _ in topk:
            scores[lb] = scores.get(lb, 0.0) + sim

        best_label = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_label] / total_score if total_score > 0 else 0.0

        sem_th = self.semantic_thresholds().get(best_label, 0.70)
        if confidence >= sem_th:
            prob_map = {x: 0.0 for x in LABELS}
            prob_map[best_label] = confidence
            return {
                "stage": "SEMANTIC",
                "label": best_label,
                "confidence": confidence,
                "reason": f"semantic {best_label} conf={confidence:.4f} >= th={sem_th:.2f}",
                "probabilities": prob_map,
                "topk": topk,
            }

        # 3. ML fallback
        ml_label, ml_conf, prob_map = self.ml_predict(text)
        ml_th = self.ml_thresholds().get(ml_label, 0.55)
        if ml_conf >= ml_th:
            return {
                "stage": "ML",
                "label": ml_label,
                "confidence": ml_conf,
                "reason": f"ml {ml_label} conf={ml_conf:.4f} >= th={ml_th:.2f}",
                "probabilities": prob_map,
                "topk": topk,
            }

        # 4. UNCERTAIN fallback
        return {
            "stage": "UNCERTAIN",
            "label": "UNCERTAIN",
            "confidence": max(sem_sim, ml_conf),
            "reason": f"semantic({sem_label}:{sem_sim:.4f}<{sem_th:.2f}) & ml({ml_label}:{ml_conf:.4f}<{ml_th:.2f})",
            "probabilities": prob_map,
            "topk": topk,
        }

# -----------------------------
# 白名單解析
# -----------------------------
def parse_whitelist(raw: str) -> set[str]:
    """將逗號分隔的域名轉成集合"""
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return set(vals) if vals else set(WHITELIST_DOMAINS)

# -----------------------------
# 主程式
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="backend_ml_guardrail_model.joblib")
    parser.add_argument("--data", default="backend_ml_data_guardrail_augmented_big.csv")
    parser.add_argument("--embed-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--whitelist",
        default=",".join(sorted(WHITELIST_DOMAINS)),
        help="comma-separated domains, e.g. yourdomain.com,www.yourdomain.com"
    )
    args = parser.parse_args()

    # 解析白名單
    whitelist = parse_whitelist(args.whitelist)

    # 初始化 RSMTester
    tester = RSMTester(
        ml_model_path=Path(args.model),
        data_path=Path(args.data),
        whitelist_domains=whitelist,
        embedding_model_name=args.embed_model,
        semantic_topk=args.topk,
    )

    # 互動式測試
    print("=== Guardrail Interactive (Rule > Semantic > ML) ===")
    print(f"ML model: {args.model}")
    print(f"Data    : {args.data}")
    print(f"Whitelist domains: {sorted(whitelist)}")
    print("輸入 q / quit / exit 離開")

    while True:
        text = input("\n請輸入測試文字 > ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            print("bye.")
            break
        if not text:
            print("（空字串略過）")
            continue

        out = tester.classify(text)

        # 輸出結果
        print("-" * 88)
        print(f"Input       : {text}")
        print(f"Stage       : {out['stage']}")
        print(f"Prediction  : {out['label']}")
        print(f"Confidence  : {out['confidence']:.4f}")
        print(f"Reason      : {out['reason']}")
        print("Probabilities(ML or semantic score):")
        for lb in LABELS:
            print(f"  - {lb:14s}: {out['probabilities'].get(lb, 0.0):.4f}")

        if out["topk"]:
            print("Semantic Top-K Neighbors:")
            for i, (idx, sim, lb, tx) in enumerate(out["topk"], start=1):
                print(f"  {i}. sim={sim:.4f} label={lb} text={tx}")

        if out["label"] == "UNCERTAIN":
            print("Note        : 建議走 safe_fallback 或 LLM 二判")
        print("-" * 88)

# 執行入口
if __name__ == "__main__":
    main()