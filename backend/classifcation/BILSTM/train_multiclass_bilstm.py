# -*- coding: utf-8 -*-
"""字元級 BiLSTM，單模型 4 選 1。

切分 / 評估 / 輸出路徑由 common.py 統一；模型與訓練迴圈在 bilstm_lib.py（pack_padded_sequence + class weight）。
報告存 reports/bilstm_multiclass.json。
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import bilstm_lib as B  # noqa: E402

METHOD = "bilstm_multiclass"
OUT = C.OUT_PATHS[METHOD]
MODEL_PATH = OUT / "backend_ml_guardrail_bilstm_pytorch.pt"
VOCAB_PATH = OUT / "backend_ml_guardrail_vocab.joblib"
LABEL_ENCODER_PATH = OUT / "backend_ml_guardrail_label_encoder.joblib"

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
    X_hard, y_hard = C.load_hard_eval()

    le = LabelEncoder().fit(C.LABELS)
    vocab = B.build_vocab(s.X_train, MAX_VOCAB_SIZE)  # 只用 train 建 vocab
    print(f"Vocab size: {len(vocab)}")

    def loader(X, y, shuffle=False):
        return DataLoader(B.SeqDataset(X, le.transform(y), vocab, MAX_LEN), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = loader(s.X_train, s.y_train, shuffle=True)
    val_loader = loader(s.X_val, s.y_val)

    model = B.BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, len(le.classes_), NUM_LAYERS, DROPOUT,
                               vocab[B.PAD]).to(device)
    meta = {"vocab_size": len(vocab), "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "num_classes": len(le.classes_), "max_len": MAX_LEN, "pad_idx": vocab[B.PAD]}
    model = B.train_model(model, train_loader, val_loader, device,
                          C.balanced_class_weights(s.y_train, list(le.classes_)),
                          EPOCHS, LR, PATIENCE, MODEL_PATH, meta)

    def predict(X):
        dummy = np.array([C.LABELS[0]] * len(X), dtype=object)
        probs = B.predict_proba(model, loader(X, dummy), device)
        return le.inverse_transform(probs.argmax(axis=1))

    results = {"method": METHOD,
               "test": C.evaluate_split(s.X_test, s.y_test, predict(s.X_test), f"{METHOD} / test")}
    if X_hard is not None:
        results["hard_eval"] = C.evaluate_split(X_hard, y_hard, predict(X_hard), f"{METHOD} / hard_eval")

    joblib.dump(vocab, VOCAB_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Saved model => {MODEL_PATH}\nSaved vocab => {VOCAB_PATH}\nSaved label encoder => {LABEL_ENCODER_PATH}")
    C.save_report(METHOD, results)


if __name__ == "__main__":
    main()
