import json

# 讀取原始檔
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 轉換格式
new_data = [
    {
        "question": item.get("instruction", ""),
        "aliases": [],
        "answer": item.get("output", "")
    }
    for item in data
]

# 寫入新檔
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"已轉換 {len(new_data)} 筆資料")