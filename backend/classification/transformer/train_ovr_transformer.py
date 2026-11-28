# -*- coding: utf-8 -*-
"""Transformer（hfl/chinese-roberta-wwm-ext）OvR：每個 label 一個二元模型。🏆 上線模型。

輸出：transformer/backend_ml_ovr_transformer_models_clean/<label>/ + ovr_config.json
      ＝ backend/app/services/guardrail.py 讀取的位置（原本相對 CWD，重訓結果寫不到上線位置）

修正（2026-11-26）：輸出路徑固定、共用切分（common.py）、class weight、EPOCHS 5、MAX_LEN 256、
門檻平台取中位數，並以上線決策規則（過門檻取最高、都沒過 → UNCERTAIN）組合出 4 類結果評估。
報告存 reports/transformer_ovr.json。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import classification_report
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import hf_lib as H  # noqa: E402

METHOD, OUT_DIR = H.tagged("transformer_ovr", C.OUT_PATHS["transformer_ovr"])


def main() -> None:
    H.set_seed(C.SEED)
    df = C.load_dataset()
    s = C.split_dataset(df)
    evals = C.load_eval_sets()  # hard_eval + blind_eval
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(H.MODEL_NAME)

    thresholds: Dict[str, float] = {}
    binary_reports: Dict[str, str] = {}
    test_probs = []
    eval_probs: Dict[str, list] = {name: [] for name in evals}

    for lb in C.LABELS:
        print("\n" + "=" * 72)
        print(f"Training: {lb} vs NOT_{lb}")
        yb = {k: (getattr(s, f"y_{k}") == lb).astype(int) for k in ("train", "val", "test")}
        model_dir = OUT_DIR / lb.lower()
        model_dir.mkdir(parents=True, exist_ok=True)

        trainer = H.WeightedTrainer(
            model=AutoModelForSequenceClassification.from_pretrained(H.MODEL_NAME, num_labels=2),
            args=H.training_args(str(model_dir / "checkpoints"), C.SEED),
            train_dataset=H.TextDataset(s.X_train, yb["train"], tokenizer),
            eval_dataset=H.TextDataset(s.X_val, yb["val"], tokenizer),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=H.PATIENCE)],
            class_weights=C.balanced_class_weights(yb["train"], [0, 1]),
        )
        trainer.train()

        t = C.find_best_threshold(yb["val"], H.predict_proba(trainer, s.X_val, tokenizer)[:, 1], lb)
        thresholds[lb] = t
        p_test = H.predict_proba(trainer, s.X_test, tokenizer)[:, 1]
        rep = classification_report(yb["test"], (p_test >= t).astype(int), digits=4, zero_division=0)
        binary_reports[lb] = rep
        print(f"[{lb}] threshold(from val) = {t}")
        print(rep)
        test_probs.append(p_test)
        for name, (X_e, _) in evals.items():
            eval_probs[name].append(H.predict_proba(trainer, X_e, tokenizer)[:, 1])

        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

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

    C.write_ovr_config(OUT_DIR, thresholds, {
        "model_name": H.MODEL_NAME,
        "max_len": H.MAX_LEN,
        "model_dir": f"transformer/{OUT_DIR.name}",
        "note": "Threshold selected on VAL only; TEST used once for final report.",
    })
    C.save_report(METHOD, results)


if __name__ == "__main__":
    main()
