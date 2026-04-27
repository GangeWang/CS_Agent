from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]

# 你的原始資料檔（每行: 句子,標籤）
RAW_PATH = Path("classifcation/backend_ml_data_guardrail_augmented_big.csv")
MODEL_DIR = Path("semantic_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 讀取兩欄: text,label
df = pd.read_csv(
    RAW_PATH,
    header=None,                # 沒有表頭
    names=["text", "label"],    # 自訂欄位名
    encoding="utf-8"
)

# 清理
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.upper()
df = df[(df["text"] != "") & (df["label"].isin(LABELS))]
df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

texts = df["text"].tolist()
labels = df["label"].tolist()

print("rows:", len(df))
print(df["label"].value_counts())

# 產 embeddings
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
emb = embedder.encode(texts, normalize_embeddings=True)

# 存檔（你的 guardrail.py 會讀這三個）
joblib.dump(texts, MODEL_DIR / "semantic_texts.joblib")
joblib.dump(labels, MODEL_DIR / "semantic_labels.joblib")
np.save(MODEL_DIR / "semantic_embeddings.npy", emb)

print("saved:")
print("-", MODEL_DIR / "semantic_texts.joblib")
print("-", MODEL_DIR / "semantic_labels.joblib")
print("-", MODEL_DIR / "semantic_embeddings.npy")