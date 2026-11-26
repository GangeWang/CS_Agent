import pandas as pd

input_file = "../abusive_translated_zh.csv"
output_file = "full_data_set.csv"

df = pd.read_csv(input_file)

# 刪除 text 中含有英文字母的資料
df = df[~df["text"].str.contains(r"[A-Za-z]", na=False)]

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"原始資料：{len(pd.read_csv(input_file))} 筆")
print(f"剩餘資料：{len(df)} 筆")
print(f"已輸出：{output_file}")