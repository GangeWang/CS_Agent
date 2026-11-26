# -*- coding: utf-8 -*-
"""BiLSTM 兩支訓練腳本共用的 vocab / Dataset / 模型 / 訓練迴圈。

修正（2026-11-26）：
- 原本補 PAD 到 200 字後直接丟進 LSTM、沒有 pack：正向 LSTM 的最終狀態是讀完一百多個 PAD 後的狀態，
  文字訊號被稀釋；PAD 數量也等於把句長直接告訴模型（加重「長訊息 ≠ NORMAL」的長度捷徑）。
  → 改用 pack_padded_sequence，只看真實長度
- 加入 class weight（PROMPT_ATTACK 只佔約 7%，原本沒有任何不平衡處理）
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PAD, UNK = "<PAD>", "<UNK>"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    return list(str(text))  # 中文字元級


def build_vocab(texts: Sequence[str], max_vocab_size: int = 50000, min_freq: int = 1) -> Dict[str, int]:
    counter: Counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {PAD: 0, UNK: 1}
    for tok, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_vocab_size:
            break
        vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int):
    ids = [vocab.get(t, vocab[UNK]) for t in tokenize(text)][:max_len] or [vocab[UNK]]
    length = len(ids)
    return ids + [vocab[PAD]] * (max_len - length), length


class SeqDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Dict[str, int], max_len: int):
        enc = [encode(t, vocab, max_len) for t in texts]
        self.ids = torch.tensor([e[0] for e in enc], dtype=torch.long)
        self.lengths = torch.tensor([e[1] for e in enc], dtype=torch.long)
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.lengths[i], self.labels[i]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # 最後一層 forward / backward
        return self.fc(self.dropout(h))


def predict_proba(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for x, lengths, _ in loader:
            out.append(torch.softmax(model(x.to(device), lengths), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device, class_weights,
                epochs: int, lr: float, patience: int, ckpt_path: Path, ckpt_meta: dict) -> nn.Module:
    """以 val loss 做 early stopping，保存最佳權重並載回。"""
    w = torch.tensor(class_weights, dtype=torch.float, device=device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w)
    eval_criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, bad = float("inf"), 0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        train_loss = total / len(train_loader.dataset)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, y = x.to(device), y.to(device)
                vtotal += eval_criterion(model(x, lengths), y).item() * x.size(0)
        val_loss = vtotal / len(val_loader.dataset)
        print(f"Epoch [{epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best:
            best, bad = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "packed_sequence": True, **ckpt_meta}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    return model
