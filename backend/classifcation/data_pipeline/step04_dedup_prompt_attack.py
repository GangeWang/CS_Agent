import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 設定
# =========================

INPUT_FILE = "extracted_prompt_attack.csv"
OUTPUT_FILE = "deduped_prompt_attack.csv"

# 相似度門檻
# 越高 = 越嚴格，只刪除非常相似的資料
SIMILARITY_THRESHOLD = 0.8


# =========================
# 讀取 CSV
# =========================

def load_data(filename):
    rows = []

    with open(filename, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if "text" not in reader.fieldnames:
            raise ValueError("CSV 缺少 text 欄位")

        if "label" not in reader.fieldnames:
            raise ValueError("CSV 缺少 label 欄位")

        for row in reader:
            text = row["text"]

            if text is None or not text.strip():
                continue

            rows.append({
                "text": text.strip(),
                "label": row["label"]
            })

    return rows


# =========================
# 相似度去重
# =========================

def remove_similar_rows(rows, threshold):
    texts = [row["text"] for row in rows]

    print(f"原始資料：{len(texts)} 筆")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    # 計算 cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    removed = set()
    kept_rows = []

    for i in range(len(rows)):

        # 如果這筆已經被刪除，就跳過
        if i in removed:
            continue

        kept_rows.append(rows[i])

        # 找出與目前資料過於相似的後續資料
        for j in range(i + 1, len(rows)):

            if j in removed:
                continue

            similarity = similarity_matrix[i][j]

            if similarity >= threshold:
                removed.add(j)

    print(f"刪除相似資料：{len(removed)} 筆")
    print(f"保留資料：{len(kept_rows)} 筆")

    return kept_rows


# =========================
# 儲存 CSV
# =========================

def save_data(filename, rows):

    output_path = Path(filename)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["text", "label"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"輸出完成：{filename}")


# =========================
# Main
# =========================

if __name__ == "__main__":

    rows = load_data(INPUT_FILE)

    filtered_rows = remove_similar_rows(
        rows,
        SIMILARITY_THRESHOLD
    )

    save_data(
        OUTPUT_FILE,
        filtered_rows
    )
