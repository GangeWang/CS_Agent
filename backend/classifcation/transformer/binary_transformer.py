from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import json
import random
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed as hf_set_seed,
)

# ===================== 路徑與常數 =====================
DATA_PATH = Path("backend_ml_data_guardrail_augmented_big.csv")
OUT_DIR = Path("./backend_ml_ovr_transformer_models_clean")
LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]
MODEL_NAME = "bert-base-chinese"

# ===================== 可調參數 =====================
SEED = 42
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


class BinaryTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


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


def predict_pos_prob(trainer: Trainer, ds: Dataset) -> np.ndarray:
    pred_output = trainer.predict(ds)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return probs


def train_one_binary(
    target_label: str,
    tokenizer,
    X_train, y_train_bin,
    X_val, y_val_bin,
    X_test, y_test_bin,
) -> Tuple[float, str]:
    print("\n" + "=" * 72)
    print(f"Training: {target_label} vs NOT_{target_label}")

    model_out_dir = OUT_DIR / target_label.lower()
    ckpt_dir = model_out_dir / "checkpoints"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = BinaryTextDataset(X_train, y_train_bin, tokenizer, MAX_LEN)
    val_ds = BinaryTextDataset(X_val, y_val_bin, tokenizer, MAX_LEN)
    test_ds = BinaryTextDataset(X_test, y_test_bin, tokenizer, MAX_LEN)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
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
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # 1) 在 val 找 threshold
    val_prob = predict_pos_prob(trainer, val_ds)
    best_t = find_best_threshold_on_val(np.array(y_val_bin), val_prob)

    # 2) 在 test 固定該 threshold 做最終評估
    test_prob = predict_pos_prob(trainer, test_ds)
    y_pred = (test_prob >= best_t).astype(int)

    rep = classification_report(y_test_bin, y_pred, digits=4, zero_division=0)
    print(f"[{target_label}] threshold(from val) = {best_t}")
    print(rep)

    trainer.save_model(str(model_out_dir))
    tokenizer.save_pretrained(str(model_out_dir))

    return best_t, rep


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"].isin(LABELS))].reset_index(drop=True)

    print(f"Loaded rows(raw cleaned): {len(df)}")
    print(df["label"].value_counts())

    # 先切分（防止全資料先去重造成洩漏偏樂觀）
    X = df["text"].values
    y = df["label"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=SEED, stratify=y_train_full
    )

    # 只在 train 內去重（同 text+label）
    train_df = pd.DataFrame({"text": X_train, "label": y_train}).drop_duplicates(["text", "label"])
    X_train = train_df["text"].values
    y_train = train_df["label"].values

    # 洩漏檢查（文字完全重複）
    train_text_set = set(map(str, X_train))
    test_text_set = set(map(str, X_test))
    overlap = len(train_text_set & test_text_set)
    print(f"Train/Test exact text overlap: {overlap} / {len(test_text_set)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    threshold_map: Dict[str, float] = {}
    reports: Dict[str, str] = {}

    for target_label in LABELS:
        y_train_bin = (y_train == target_label).astype(int)
        y_val_bin = (y_val == target_label).astype(int)
        y_test_bin = (y_test == target_label).astype(int)

        best_t, rep = train_one_binary(
            target_label=target_label,
            tokenizer=tokenizer,
            X_train=X_train, y_train_bin=y_train_bin,
            X_val=X_val, y_val_bin=y_val_bin,
            X_test=X_test, y_test_bin=y_test_bin
        )
        threshold_map[target_label] = best_t
        reports[target_label] = rep

    config = {
        "labels": LABELS,
        "thresholds": threshold_map,
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "model_dir": str(OUT_DIR.as_posix()),
        "note": "Threshold selected on VAL only; TEST used once for final report."
    }
    with open(OUT_DIR / "ovr_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print("\nSaved =>", (OUT_DIR / "ovr_config.json").resolve())
    print("Thresholds:", threshold_map)


if __name__ == "__main__":
    main()