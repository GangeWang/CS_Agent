# -*- coding: utf-8 -*-
"""字元級 BiLSTM，OvR（每個 label 一個二元模型）。

修正（2026-11-26）：輸出路徑固定到 BILSTM/backend_ml_ovr_bilstm_models（原本相對 CWD）、
pack_padded_sequence、class weight、門檻平台取中位數，並以上線決策規則組合出 4 類結果評估。
報告存 reports/bilstm_ovr.json。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import bilstm_lib as B  # noqa: E402

METHOD = "bilstm_ovr"
OUT_DIR = C.OUT_PATHS[METHOD]

MAX_VOCAB_SIZE = 50000
MAX_LEN = 200
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
PATIENCE = 3


def main() -> None:
    B.set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = C.load_dataset()
    s = C.split_dataset(df)
    evals = C.load_eval_sets()  # hard_eval + blind_eval

    vocab = B.build_vocab(s.X_train, MAX_VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocab, OUT_DIR / "vocab.joblib")

    def loader(X, y_bin, shuffle=False):
        return DataLoader(B.SeqDataset(X, y_bin, vocab, MAX_LEN), batch_size=BATCH_SIZE, shuffle=shuffle)

    thresholds: Dict[str, float] = {}
    binary_reports: Dict[str, str] = {}
    test_probs = []
    eval_probs: Dict[str, list] = {name: [] for name in evals}

    for lb in C.LABELS:
        print("\n" + "=" * 72)
        print(f"Training binary model: {lb} vs NOT_{lb}")
        yb = {k: (getattr(s, f"y_{k}") == lb).astype(int) for k in ("train", "val", "test")}

        model = B.BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, 2, NUM_LAYERS, DROPOUT, vocab[B.PAD]).to(device)
        meta = {"vocab_size": len(vocab), "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS, "dropout": DROPOUT, "num_classes": 2, "max_len": MAX_LEN,
                "pad_idx": vocab[B.PAD]}
        ckpt = OUT_DIR / lb.lower() / f"{lb.lower()}_bilstm.pt"
        model = B.train_model(model, loader(s.X_train, yb["train"], True), loader(s.X_val, yb["val"]), device,
                              C.balanced_class_weights(yb["train"], [0, 1]), EPOCHS, LR, PATIENCE, ckpt, meta)

        t = C.find_best_threshold(yb["val"], B.predict_proba(model, loader(s.X_val, yb["val"]), device)[:, 1], lb)
        thresholds[lb] = t
        p_test = B.predict_proba(model, loader(s.X_test, yb["test"]), device)[:, 1]
        rep = classification_report(yb["test"], (p_test >= t).astype(int), digits=4, zero_division=0)
        binary_reports[lb] = rep
        print(f"[{lb}] threshold(from val) = {t}")
        print(rep)
        test_probs.append(p_test)
        for name, (X_e, _) in evals.items():
            eval_probs[name].append(B.predict_proba(model, loader(X_e, np.zeros(len(X_e), dtype=int)), device)[:, 1])
        print(f"saved => {ckpt}")

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

    C.write_ovr_config(OUT_DIR, thresholds, {"max_len": MAX_LEN, "model_dir": "BILSTM/backend_ml_ovr_bilstm_models",
                                             "note": "OvR BiLSTM (packed); thresholds selected on VAL."})
    C.save_report(METHOD, results)


if __name__ == "__main__":
    main()
