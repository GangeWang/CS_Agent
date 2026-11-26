# -*- coding: utf-8 -*-
"""Transformer 兩支訓練腳本共用的 Dataset / 加權 Trainer / 推論工具與超參數。

修正（2026-11-26）：
- 4選1 原本用 bert-base-chinese / 5 epoch / 256，OvR 用 chinese-roberta-wwm-ext / 3 epoch / 512，
  兩者無法比較 → 統一使用下方超參數
- OvR 原本 EPOCHS=3 搭配 patience=2，early stopping 形同虛設 → EPOCHS=5
- MAX_LEN 統一 256（與上線 ovr_config.json 一致；512 會讓上線推論變慢且訓練資料 99% 在 160 字內）
- 加入 class weight（加權 cross entropy）
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import set_seed as hf_set_seed

DEFAULT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
# 換預訓練模型做 A/B：GUARDRAIL_BASE_MODEL=ckiplab/bert-base-chinese python transformer/train_ovr_transformer.py
# 非預設模型時，輸出目錄與報告名稱會加上模型標籤，不會覆蓋上線模型
MODEL_NAME = os.environ.get("GUARDRAIL_BASE_MODEL", "").strip() or DEFAULT_MODEL_NAME
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 5
PATIENCE = 2
LR = 2e-5
WEIGHT_DECAY = 0.01


def model_tag() -> str:
    """預設模型回空字串；否則回 "__<模型名>"（例：__ckiplab-bert-base-chinese）。"""
    if MODEL_NAME == DEFAULT_MODEL_NAME:
        return ""
    return "__" + re.sub(r"[^A-Za-z0-9]+", "-", MODEL_NAME).strip("-")


def tagged(method: str, out_dir: Path):
    """依 model_tag() 回傳 (報告名稱, 輸出目錄)。"""
    tag = model_tag()
    return method + tag, (out_dir.with_name(out_dir.name + tag) if tag else out_dir)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


class TextDataset(Dataset):
    """動態 padding（交給 DataCollatorWithPadding），不預先補到 MAX_LEN。"""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_len: int = MAX_LEN):
        self.texts = [str(t) for t in texts]
        self.labels = [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = self.labels[idx]
        return enc


class WeightedTrainer(Trainer):
    """以 class weight 計算 cross entropy 的 Trainer。"""

    def __init__(self, *args, class_weights: Optional[Sequence[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = (torch.tensor(list(class_weights), dtype=torch.float)
                               if class_weights is not None else None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self._class_weights.to(logits.device) if self._class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def training_args(output_dir: str, seed: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
        disable_tqdm=True,
    )


def predict_proba(trainer: Trainer, texts: Sequence[str], tokenizer) -> np.ndarray:
    ds = TextDataset(texts, [0] * len(texts), tokenizer)
    logits = trainer.predict(ds).predictions
    return torch.softmax(torch.tensor(logits), dim=1).numpy()
