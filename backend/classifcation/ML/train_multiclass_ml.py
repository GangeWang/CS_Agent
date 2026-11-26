# -*- coding: utf-8 -*-
"""TF-IDF(char 1-5) + LogisticRegression，單模型 4 選 1。

資料切分 / 評估 / 輸出路徑皆由 common.py 統一（六種方法可直接比較），報告存 reports/ml_multiclass.json。
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

METHOD = "ml_multiclass"
MODEL_PATH = C.OUT_PATHS[METHOD]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=1, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=4000, class_weight="balanced", C=2.0, solver="lbfgs")),
    ])


def main() -> None:
    df = C.load_dataset()
    s = C.split_dataset(df)

    clf = build_pipeline()
    clf.fit(s.X_train, s.y_train)  # val 不參與訓練（與其他方法一致）

    results = {
        "method": METHOD,
        "test": C.evaluate_split(s.X_test, s.y_test, clf.predict(s.X_test), f"{METHOD} / test"),
    }
    X_hard, y_hard = C.load_hard_eval()
    if X_hard is not None:
        results["hard_eval"] = C.evaluate_split(X_hard, y_hard, clf.predict(X_hard), f"{METHOD} / hard_eval")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model => {MODEL_PATH}")
    C.save_report(METHOD, results)


if __name__ == "__main__":
    main()
