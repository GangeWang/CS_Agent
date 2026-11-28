# -*- coding: utf-8 -*-
"""TF-IDF(char 1-5) + LogisticRegression，OvR（每個 label 一個二元模型）。

修正（2026-11-26）：
- 門檻原本在 test 上挑選（分數偏樂觀）→ 改在 val 上挑，test 只用來報告
- 除了 4 個二元報告，另以上線決策規則（common.ovr_decide）組合出 4 類結果再評估
資料切分 / 輸出路徑由 common.py 統一，報告存 reports/ml_ovr.json。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

METHOD = "ml_ovr"
OUT_DIR = C.OUT_PATHS[METHOD]


def build_binary_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=1, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=4000, class_weight="balanced", C=2.0, solver="lbfgs")),
    ])


def main() -> None:
    df = C.load_dataset()
    s = C.split_dataset(df)
    evals = C.load_eval_sets()  # hard_eval + blind_eval
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    thresholds: Dict[str, float] = {}
    binary_reports: Dict[str, str] = {}
    test_probs = []
    eval_probs: Dict[str, list] = {name: [] for name in evals}

    for lb in C.LABELS:
        print("\n" + "=" * 72)
        print(f"Training binary model: {lb} vs NOT_{lb}")
        clf = build_binary_pipeline()
        clf.fit(s.X_train, (s.y_train == lb).astype(int))

        t = C.find_best_threshold((s.y_val == lb).astype(int), clf.predict_proba(s.X_val)[:, 1], lb)
        thresholds[lb] = t

        p_test = clf.predict_proba(s.X_test)[:, 1]
        rep = classification_report((s.y_test == lb).astype(int), (p_test >= t).astype(int), digits=4, zero_division=0)
        binary_reports[lb] = rep
        print(f"[{lb}] threshold(from val) = {t}")
        print(rep)
        test_probs.append(p_test)
        for name, (X_e, _) in evals.items():
            eval_probs[name].append(clf.predict_proba(X_e)[:, 1])

        path = OUT_DIR / f"{lb.lower()}_bin.joblib"
        joblib.dump(clf, path)
        print(f"saved => {path}")

    results = {
        "method": METHOD,
        "thresholds": thresholds,
        "binary_reports": binary_reports,
        "test": C.evaluate_split(s.X_test, s.y_test, C.ovr_decide(np.column_stack(test_probs), thresholds),
                                 f"{METHOD} / test (combined decision)"),
    }
    for name, (X_e, y_e) in evals.items():
        results[name] = C.evaluate_split(
            X_e, y_e, C.ovr_decide(np.column_stack(eval_probs[name]), thresholds),
            f"{METHOD} / {name} (combined decision)")

    C.write_ovr_config(OUT_DIR, thresholds, {"model_dir": "ML/backend_ml_ovr_models",
                                             "note": "OvR binary models; thresholds selected on VAL."})
    C.save_report(METHOD, results)


if __name__ == "__main__":
    main()
