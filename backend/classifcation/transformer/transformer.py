from pathlib import Path
import random
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed as hf_set_seed
)


# ===================== 路徑設定 =====================
DATA_PATH = Path("backend_ml_data_guardrail_augmented_big.csv")
OUTPUT_DIR = Path("backend_ml_guardrail_transformer")  # 存模型與tokenizer
LABEL_ENCODER_PATH = OUTPUT_DIR / "label_encoder.joblib"

# ===================== 可調參數 =====================
SEED = 42
MODEL_NAME = "bert-base-chinese"   # 中文建議；多語可改 xlm-roberta-base
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


class GuardrailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
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
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}, got: {list(df.columns)}")

    # ===================== 資料清理 =====================
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
        X_train_text, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )

    # tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes
    )

    train_ds = GuardrailDataset(X_train_text, y_train, tokenizer, MAX_LEN)
    val_ds = GuardrailDataset(X_val_text, y_val, tokenizer, MAX_LEN)
    test_ds = GuardrailDataset(X_test_text, y_test, tokenizer, MAX_LEN)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # HuggingFace Trainer 參數
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
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
        seed=SEED
    )

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    # ===================== 訓練 =====================
    trainer.train()

    # ===================== 測試評估 =====================
    pred_output = trainer.predict(test_ds)
    y_pred = np.argmax(pred_output.predictions, axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=4, zero_division=0
    ))

    # ===================== 儲存 =====================
    # 儲存 best model + tokenizer
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    joblib.dump(le, LABEL_ENCODER_PATH)

    print(f"\nSaved model/tokenizer => {OUTPUT_DIR.resolve()}")
    print(f"Saved label encoder => {LABEL_ENCODER_PATH.resolve()}")


if __name__ == "__main__":
    main()