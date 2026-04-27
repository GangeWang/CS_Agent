from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


DATA_PATH = Path("backend_ml_data_guardrail_augmented_big.csv")
OUT_DIR = Path("./backend_ml_ovr_models")
LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]


@dataclass
class BinModelResult:
    label: str
    threshold: float
    report: str


def build_binary_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 5),
            min_df=1,
            sublinear_tf=True
        )),
        ("lr", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=2.0,
            solver="lbfgs"
        ))
    ])


def find_best_threshold(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    # 用 F1 掃 threshold（可改成偏 recall 或 precision）
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.91, 0.02):
        y_pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(round(t, 2))
    return best_t


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(LABELS)]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    print(f"Loaded rows: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold_map: Dict[str, float] = {}
    reports: Dict[str, str] = {}

    # 訓練四個二元模型
    for target_label in LABELS:
        print("\n" + "=" * 72)
        print(f"Training binary model: {target_label} vs NOT_{target_label}")

        y_train_bin = (y_train == target_label).astype(int)
        y_test_bin = (y_test == target_label).astype(int)

        clf = build_binary_pipeline()
        clf.fit(X_train, y_train_bin)

        # 機率（正類 = 1）
        y_prob = clf.predict_proba(X_test)[:, 1]
        best_t = find_best_threshold(y_test_bin, y_prob)
        threshold_map[target_label] = best_t

        y_pred = (y_prob >= best_t).astype(int)
        rep = classification_report(y_test_bin, y_pred, digits=4, zero_division=0)
        reports[target_label] = rep

        print(f"[{target_label}] best threshold = {best_t}")
        print(rep)

        model_path = OUT_DIR / f"{target_label.lower()}_bin.joblib"
        joblib.dump(clf, model_path)
        print(f"saved => {model_path.resolve()}")

    # threshold 設定
    config = {
        "labels": LABELS,
        "thresholds": threshold_map,
        "model_dir": str(OUT_DIR.as_posix()),
        "note": "OvR binary models: label vs not_label"
    }
    with open(OUT_DIR / "ovr_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Saved config =>", (OUT_DIR / "ovr_config.json").resolve())
    print("Thresholds:", threshold_map)


if __name__ == "__main__":
    main()