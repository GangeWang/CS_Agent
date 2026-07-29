import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ======================
# 參數設定
# ======================
INPUT_FILE = "ML_data.csv"
OUTPUT_FILE = "data_set/clean.csv"
SIMILARITY_THRESHOLD = 0.95

# ======================
# 讀取資料
# ======================
df = pd.read_csv(INPUT_FILE)

print(f"原始資料數：{len(df)}")

# ======================
# 載入模型
# ======================
print("Loading model...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ======================
# 建立 Embedding
# ======================
print("Encoding sentences...")

embeddings = model.encode(
    df["text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

# ======================
# 計算 Cosine Similarity
# ======================
print("Calculating similarity...")

similarity_matrix = cosine_similarity(embeddings)

# ======================
# 找出重複
# ======================
duplicates = set()

print("\nDuplicate Pairs:\n")

for i in tqdm(range(len(df))):

    if i in duplicates:
        continue

    for j in range(i + 1, len(df)):

        if j in duplicates:
            continue

        score = similarity_matrix[i][j]

        if score >= SIMILARITY_THRESHOLD:

            duplicates.add(j)

            print("=" * 80)
            print(f"Similarity: {score:.4f}")
            print(f"Keep   [{i}] ({df.iloc[i]['label']}):")
            print(df.iloc[i]["text"])
            print()
            print(f"Remove [{j}] ({df.iloc[j]['label']}):")
            print(df.iloc[j]["text"])
            print()

# ======================
# 去重
# ======================
df_clean = df.drop(index=list(duplicates))

print("=" * 80)
print(f"原始資料：{len(df)}")
print(f"刪除重複：{len(duplicates)}")
print(f"剩餘資料：{len(df_clean)}")

# ======================
# 儲存
# ======================
df_clean.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n已輸出：{OUTPUT_FILE}")