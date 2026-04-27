from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import json
import random
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===================== 路徑設定 =====================
DATA_PATH = Path("backend_ml_data_guardrail_augmented_big.csv")
OUT_DIR = Path("./backend_ml_ovr_bilstm_models")
LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]

# ===================== 可調參數 =====================
SEED = 42
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str):
    # 中文建議字元級
    return list(text)


def build_vocab(texts, max_vocab_size=50000, min_freq=1):
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(str(t)))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab, max_len=200):
    tokens = simple_tokenize(str(text))
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2, num_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h = torch.cat([h_forward, h_backward], dim=1)
        h = self.dropout(h)
        logits = self.fc(h)
        return logits


def evaluate_loss(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def predict_pos_prob(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())
    return np.array(probs)


def find_best_threshold_on_val(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.91, 0.02):
        y_pred = (y_prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(round(t, 2))
    return best_t


def train_one_binary(
    target_label: str,
    X_train, y_train_bin,
    X_val, y_val_bin,
    X_test, y_test_bin,
    vocab,
    device
) -> Tuple[float, str]:
    print("\n" + "=" * 72)
    print(f"Training binary model: {target_label} vs NOT_{target_label}")

    model_dir = OUT_DIR / target_label.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{target_label.lower()}_bilstm.pt"

    # encode
    X_train_seq = [encode_text(t, vocab, MAX_LEN) for t in X_train]
    X_val_seq = [encode_text(t, vocab, MAX_LEN) for t in X_val]
    X_test_seq = [encode_text(t, vocab, MAX_LEN) for t in X_test]

    train_ds = TextDataset(X_train_seq, y_train_bin)
    val_ds = TextDataset(X_val_seq, y_val_bin)
    test_ds = TextDataset(X_test_seq, y_test_bin)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab["<PAD>"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_loss(model, val_loader, device, criterion)
        print(f"Epoch [{epoch}/{EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "num_classes": 2,
                "max_len": MAX_LEN,
                "pad_idx": vocab["<PAD>"]
            }, model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # threshold on val
    val_prob = predict_pos_prob(model, val_loader, device)
    best_t = find_best_threshold_on_val(np.array(y_val_bin), val_prob)

    # final eval on test
    test_prob = predict_pos_prob(model, test_loader, device)
    y_pred = (test_prob >= best_t).astype(int)

    rep = classification_report(y_test_bin, y_pred, digits=4, zero_division=0)
    print(f"[{target_label}] threshold(from val) = {best_t}")
    print(rep)
    print(f"saved => {model_path.resolve()}")

    return best_t, rep


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"].isin(LABELS))].reset_index(drop=True)

    print(f"Loaded rows(raw cleaned): {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())

    X = df["text"].values
    y = df["label"].values

    # 先切分，避免洩漏偏樂觀
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
    )

    # train-only 去重
    train_df = pd.DataFrame({"text": X_train, "label": y_train}).drop_duplicates(["text", "label"])
    X_train = train_df["text"].values
    y_train = train_df["label"].values

    # overlap check
    overlap = len(set(map(str, X_train)) & set(map(str, X_test)))
    print(f"Train/Test exact text overlap: {overlap}")

    # 建 vocab（只用 train）
    vocab = build_vocab(X_train, max_vocab_size=MAX_VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vocab, OUT_DIR / "vocab.joblib")

    threshold_map: Dict[str, float] = {}
    reports: Dict[str, str] = {}

    for target_label in LABELS:
        y_train_bin = (y_train == target_label).astype(int)
        y_val_bin = (y_val == target_label).astype(int)
        y_test_bin = (y_test == target_label).astype(int)

        best_t, rep = train_one_binary(
            target_label=target_label,
            X_train=X_train, y_train_bin=y_train_bin,
            X_val=X_val, y_val_bin=y_val_bin,
            X_test=X_test, y_test_bin=y_test_bin,
            vocab=vocab,
            device=device
        )
        threshold_map[target_label] = best_t
        reports[target_label] = rep

    config = {
        "labels": LABELS,
        "thresholds": threshold_map,
        "max_len": MAX_LEN,
        "model_dir": str(OUT_DIR.as_posix()),
        "note": "OvR BiLSTM binary models: label vs not_label; threshold selected on VAL only."
    }

    with open(OUT_DIR / "ovr_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Saved config =>", (OUT_DIR / "ovr_config.json").resolve())
    print("Thresholds:", threshold_map)


if __name__ == "__main__":
    main()