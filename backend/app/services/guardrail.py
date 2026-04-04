"""
Guardrail text classification service.
Loads OvR binary models from backend_ml_ovr_models and provides inference API.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple
import json
import logging

import joblib

logger = logging.getLogger(__name__)

LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]
MODEL_DIR = Path(__file__).resolve().parents[2] / "backend_ml_ovr_models"
CONFIG_PATH = MODEL_DIR / "ovr_config.json"
ModelMap = Dict[str, Any]
ThresholdMap = Dict[str, float]


def _predict_ovr(
        models: ModelMap,
        thresholds: ThresholdMap,
        text: str
) -> Tuple[str, Dict[str, float], Dict[str, bool]]:
    probs: Dict[str, float] = {}
    pass_flags: Dict[str, bool] = {}

    for lb in LABELS:
        prob_pos = float(models[lb].predict_proba([text])[0, 1])
        probs[lb] = prob_pos
        pass_flags[lb] = prob_pos >= float(thresholds[lb])

    passed = [lb for lb in LABELS if pass_flags[lb]]
    if passed:
        best = max(passed, key=lambda x: probs[x])
        return best, probs, pass_flags
    return "UNCERTAIN", probs, pass_flags


@lru_cache(maxsize=1)
def _load_guardrail_models() -> Tuple[ModelMap, ThresholdMap]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = cfg["thresholds"]

    models = {}
    for lb in LABELS:
        p = MODEL_DIR / f"{lb.lower()}_bin.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Model missing: {p}")
        models[lb] = joblib.load(p)
    return models, thresholds


def classify_text(text: str) -> dict:
    """
    Classify text using OvR binary models.
    Returns fallback-safe NORMAL when models are unavailable.
    """
    try:
        models, thresholds = _load_guardrail_models()
        pred, probs, flags = _predict_ovr(models, thresholds, text)
        return {"label": pred, "probs": probs, "flags": flags, "available": True}
    except Exception as e:
        logger.warning(f"Guardrail classification unavailable: {e}")
        return {"label": "NORMAL", "probs": {}, "flags": {}, "available": False}
