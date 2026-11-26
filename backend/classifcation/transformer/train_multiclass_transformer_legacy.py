# -*- coding: utf-8 -*-
"""Transformer 單模型 4 選 1（與 OvR 使用相同預訓練模型 / 超參數 / 切分，可直接比較）。

修正（2026-11-26）：原本用 bert-base-chinese、5 epoch、與 OvR 不同的測試集，比較結果不具意義；
輸出路徑原本相對 CWD。現在共用 common.py 與 hf_lib.py，報告存 reports/transformer_multiclass.json。
推論成本只有 OvR 的 1/4（一次 forward vs 四次）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import hf_lib as H  # noqa: E402

METHOD, OUT_DIR = H.tagged("transformer_multiclass", C.OUT_PATHS["transformer_multiclass"])
LABEL_ENCODER_PATH = OUT_DIR / "label_encoder.joblib"


def main() -> None:
    H.set_seed(C.SEED)
    df = C.load_dataset()
    s = C.split_dataset(df)
    X_hard, y_hard = C.load_hard_eval()

    le = LabelEncoder().fit(C.LABELS)
    tokenizer = AutoTokenizer.from_pretrained(H.MODEL_NAME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = H.WeightedTrainer(
        model=AutoModelForSequenceClassification.from_pretrained(H.MODEL_NAME, num_labels=len(le.classes_)),
        args=H.training_args(str(OUT_DIR / "checkpoints"), C.SEED),
        train_dataset=H.TextDataset(s.X_train, le.transform(s.y_train), tokenizer),
        eval_dataset=H.TextDataset(s.X_val, le.transform(s.y_val), tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=H.PATIENCE)],
        class_weights=C.balanced_class_weights(s.y_train, list(le.classes_)),
    )
    trainer.train()

    def predict(X):
        return le.inverse_transform(H.predict_proba(trainer, X, tokenizer).argmax(axis=1))

    results = {"method": METHOD,
               "test": C.evaluate_split(s.X_test, s.y_test, predict(s.X_test), f"{METHOD} / test")}
    if X_hard is not None:
        results["hard_eval"] = C.evaluate_split(X_hard, y_hard, predict(X_hard), f"{METHOD} / hard_eval")

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\nSaved model/tokenizer => {OUT_DIR}\nSaved label encoder => {LABEL_ENCODER_PATH}")
    C.save_report(METHOD, results)


if __name__ == "__main__":
    main()
