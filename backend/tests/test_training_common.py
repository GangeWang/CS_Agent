"""classification/common.py：六種訓練方式共用的切分 / 門檻 / 決策邏輯（純 numpy/pandas/sklearn，不需模型）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CLS_DIR = Path(__file__).resolve().parents[1] / "classification"
sys.path.insert(0, str(CLS_DIR))

C = pytest.importorskip("common")
pytest.importorskip("sklearn")


def test_out_paths_anchored_and_match_production():
    for p in C.OUT_PATHS.values():
        assert p.is_absolute()
    # 上線 guardrail 讀取的位置
    from app.services import guardrail  # noqa: F401  (torch 以 stub 取代時也可 import)
    assert C.OUT_PATHS["transformer_ovr"].resolve() == guardrail.MODEL_DIR.resolve()


def test_ovr_decide_matches_production_rule():
    th = {"NORMAL": 0.5, "ABUSIVE": 0.5, "PROMPT_ATTACK": 0.5, "SPAM": 0.5}
    probs = np.array([
        [0.9, 0.6, 0.1, 0.1],   # 兩個過門檻 → 取最高 NORMAL
        [0.2, 0.3, 0.1, 0.4],   # 都沒過 → UNCERTAIN（即使 SPAM 最高）
        [0.1, 0.2, 0.7, 0.8],   # SPAM
    ])
    assert list(C.ovr_decide(probs, th)) == ["NORMAL", "UNCERTAIN", "SPAM"]


def test_threshold_plateau_takes_median_not_minimum():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])  # 0.05~0.95 全部 F1=1 → 舊版會取下限
    t = C.find_best_threshold(y, p, "X")
    assert 0.45 <= t <= 0.55


def test_split_is_deterministic_and_leak_free():
    df = C.load_dataset(verbose=False)
    a = C.split_dataset(df, verbose=False)
    b = C.split_dataset(df, verbose=False)
    assert list(a.X_test) == list(b.X_test)
    assert not ({C.norm_key(t) for t in a.X_train} & {C.norm_key(t) for t in a.X_test})
    assert set(a.y_test) == set(C.LABELS)


def test_hard_eval_not_in_training_data():
    X_hard, _ = C.load_hard_eval()
    df = C.load_dataset(verbose=False)
    assert X_hard is not None
    assert not ({C.norm_key(t) for t in X_hard} & set(df["key"]))
