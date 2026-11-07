import pandas as pd
import re

INPUT_PARQUET = r"E:\CS_Agent\backend\classifcation\train-00000-of-00001-036257a1398b1fca.parquet"
OUTPUT_CSV = r"E:\CS_Agent\backend\classifcation\extracted_abusive.csv"

# 手動指定欄位（依你資料）
LABEL_COL = "labels"
TEXT_COL = "comment_clean"

df = pd.read_parquet(INPUT_PARQUET)
print("columns:", df.columns.tolist())
print(df[[LABEL_COL, TEXT_COL]].head(3))

# 只取 abusive (label=1)
abusive = df[df[LABEL_COL] == 1].copy()

# 清理
abusive[TEXT_COL] = abusive[TEXT_COL].astype(str).str.strip()
abusive = abusive[abusive[TEXT_COL].str.len() >= 3]

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

abusive["norm_text"] = abusive[TEXT_COL].apply(normalize_text)
abusive = abusive.drop_duplicates(subset=["norm_text"])

# 轉成你要的格式
out = abusive[[TEXT_COL]].rename(columns={TEXT_COL: "text"})
out["label"] = "ABUSIVE"

out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"saved: {OUTPUT_CSV}, rows: {len(out)}")