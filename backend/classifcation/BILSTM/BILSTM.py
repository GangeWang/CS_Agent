from pathlib import Path
import pandas as pd
import numpy as np
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===================== 路徑設定 =====================
DATA_PATH = Path("backend_ml_data_guardrail_augmented_big.csv")
MODEL_PATH = Path("backend_ml_guardrail_bilstm_pytorch.pt")
TOKENIZER_PATH = Path("backend_ml_guardrail_vocab.joblib")
LABEL_ENCODER_PATH = Path("backend_ml_guardrail_label_encoder.joblib")


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
VAL_SPLIT = 0.1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def simple_tokenize(text: str):
    # 簡單版：用空白切詞；若你是中文且沒空白，可改成 list(text) 做字元級
    return list(text)


def build_vocab(texts, max_vocab_size=50000, min_freq=1):
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    # 保留特殊 token
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab, max_len=200):
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3, pad_idx=0):
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
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # bidirectional => *2

    def forward(self, x):
        # x: [B, L]
        emb = self.embedding(x)                # [B, L, E]
        output, (h_n, c_n) = self.lstm(emb)   # h_n: [num_layers*2, B, H]

        # 取最後一層的 forward/backward hidden state
        h_forward = h_n[-2, :, :]   # [B, H]
        h_backward = h_n[-1, :, :]  # [B, H]
        h = torch.cat([h_forward, h_backward], dim=1)  # [B, 2H]

        h = self.dropout(h)
        logits = self.fc(h)         # [B, num_classes]
        return logits


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_preds


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}, got: {list(df.columns)}")

    # 資料清理
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    print(f"Loaded rows: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())

    X = df["text"].tolist()
    y = df["label"].tolist()

    # label encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Num classes: {num_classes}, classes={list(le.classes_)}")

    # train/test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )

    # 再從 train 切一份 validation
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text, y_train, test_size=VAL_SPLIT, random_state=SEED, stratify=y_train
    )

    # vocab（只用訓練集建）
    vocab = build_vocab(X_train_text, max_vocab_size=MAX_VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")

    # encode
    X_train_seq = [encode_text(t, vocab, MAX_LEN) for t in X_train_text]
    X_val_seq = [encode_text(t, vocab, MAX_LEN) for t in X_val_text]
    X_test_seq = [encode_text(t, vocab, MAX_LEN) for t in X_test_text]

    # dataloader
    train_ds = TextDataset(X_train_seq, y_train)
    val_ds = TextDataset(X_val_seq, y_val)
    test_ds = TextDataset(X_test_seq, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab["<PAD>"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # train loop
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
        val_loss, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch}/{EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "num_classes": num_classes,
                "max_len": MAX_LEN,
                "pad_idx": vocab["<PAD>"]
            }, MODEL_PATH)

    print(f"\nBest model saved => {MODEL_PATH.resolve()}")

    # 載入最佳模型做測試評估
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\nTest loss: {test_loss:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        digits=4, zero_division=0
    ))

    # 儲存 vocab + label encoder
    joblib.dump(vocab, TOKENIZER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Saved vocab => {TOKENIZER_PATH.resolve()}")
    print(f"Saved label encoder => {LABEL_ENCODER_PATH.resolve()}")


if __name__ == "__main__":
    main()