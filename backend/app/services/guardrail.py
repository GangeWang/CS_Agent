"""
Guardrail text classification service.
Three-stage pipeline: RULE -> SEMANTIC -> ML(OvR).
Loads OvR binary models from semantic＿models and provides inference API.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse
from collections import defaultdict
import json
import logging
import re

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =============================
# Labels / Paths / Types
# =============================
LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]

MODEL_DIR = Path(__file__).resolve().parents[2] / "semantic＿models"
CONFIG_PATH = MODEL_DIR / "ovr_config.json"

# Optional semantic resources (you can replace file names by your own)
SEMANTIC_TEXTS_PATH = MODEL_DIR / "semantic_texts.joblib"       # List[str]
SEMANTIC_LABELS_PATH = MODEL_DIR / "semantic_labels.joblib"     # List[str]
SEMANTIC_EMB_PATH = MODEL_DIR / "semantic_embeddings.npy"       # np.ndarray shape=(N, D)

ModelMap = Dict[str, Any]
ThresholdMap = Dict[str, float]

# =============================
# Whitelist & Rules
# =============================
WHITELIST_DOMAINS = {
    "example.com",
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
    r"看片.*(找我|私訊|加我)",
    r"(高清|無碼|外流).*(影片|資源)?",
    r"(成人|AV片|自拍偷拍)",
    r"(私訊|加我).*(line|wechat|tg)",
]

SPAM_INTENT_PATTERNS = [
    r"加我(好友|line|wechat|tg)",
    r"私訊我",
    r"立即領獎|免費領獎|點我領",
    r"保證獲利|躺賺|日入",
    r"成人|無碼|外流|AV片|自拍偷拍",
    r"買粉|買讚|刷評價",
    r"博弈|代儲|代充",
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


def rule_first(text: str, whitelist_domains: set[str]) -> Optional[Tuple[str, float, str]]:
    t = text or ""

    for p in PROMPT_ATTACK_RULES:
        if re.search(p, t, re.IGNORECASE):
            return "PROMPT_ATTACK", 0.97, f"rule:{p}"

    for p in ABUSIVE_RULES:
        if re.search(p, t, re.IGNORECASE):
            return "ABUSIVE", 0.93, f"rule:{p}"

    spam_hit = spam_rule_with_url_context(t, whitelist_domains)
    if spam_hit:
        return spam_hit

    return None


# =============================
# ML (OvR) stage
# =============================
def _predict_ovr(
    models: ModelMap,
    thresholds: ThresholdMap,
    text: str
) -> Tuple[str, float, Dict[str, float], Dict[str, bool]]:
    probs: Dict[str, float] = {}
    pass_flags: Dict[str, bool] = {}

    for lb in LABELS:
        prob_pos = float(models[lb].predict_proba([text])[0, 1])
        probs[lb] = prob_pos
        pass_flags[lb] = prob_pos >= float(thresholds[lb])

    passed = [lb for lb in LABELS if pass_flags[lb]]
    if passed:
        best = max(passed, key=lambda x: probs[x])
        return best, probs[best], probs, pass_flags

    # none passed threshold
    best_any = max(LABELS, key=lambda x: probs[x])
    return "UNCERTAIN", probs[best_any], probs, pass_flags


# =============================
# Load resources
# =============================
@lru_cache(maxsize=1)
def _load_guardrail_resources():
    # 1) OVR config + models
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds: ThresholdMap = cfg["thresholds"]

    models: ModelMap = {}
    for lb in LABELS:
        p = MODEL_DIR / f"{lb.lower()}_bin.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Model missing: {p}")
        models[lb] = joblib.load(p)

    # 2) Semantic artifacts (optional but expected for stage-2)
    # If missing, semantic stage will be skipped gracefully.
    embedder = None
    corpus_texts: List[str] = []
    corpus_labels: List[str] = []
    corpus_emb: Optional[np.ndarray] = None
    semantic_topk = int(cfg.get("semantic_topk", 5))
    embedding_model_name = cfg.get("embedding_model_name", "paraphrase-multilingual-MiniLM-L12-v2")

    try:
        if SEMANTIC_TEXTS_PATH.exists() and SEMANTIC_LABELS_PATH.exists() and SEMANTIC_EMB_PATH.exists():
            corpus_texts = joblib.load(SEMANTIC_TEXTS_PATH)
            corpus_labels = joblib.load(SEMANTIC_LABELS_PATH)
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
        "thresholds": thresholds,
        "embedder": embedder,
        "corpus_texts": corpus_texts,
        "corpus_labels": corpus_labels,
        "corpus_emb": corpus_emb,
        "semantic_topk": semantic_topk,
        "whitelist_domains": whitelist,
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
    3) ML OVR
    Fallback-safe NORMAL only when whole service unavailable.
    """
    try:
        rs = _load_guardrail_resources()
        models = rs["models"]
        thresholds = rs["thresholds"]
        embedder = rs["embedder"]
        corpus_texts = rs["corpus_texts"]
        corpus_labels = rs["corpus_labels"]
        corpus_emb = rs["corpus_emb"]
        semantic_topk = rs["semantic_topk"]
        whitelist_domains = rs["whitelist_domains"]

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

        # Stage 3: ML (OvR)
        ml_label, ml_conf, probs, flags = _predict_ovr(models, thresholds, text)

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