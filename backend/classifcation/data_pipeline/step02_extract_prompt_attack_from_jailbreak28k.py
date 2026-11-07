import csv
from pathlib import Path


def extract_prompt_attacks(input_file, output_file):
    """
    從原始 CSV 提取 jailbreak_query，
    並建立 prompt attack 資料。

    Output:
        text,label

    label = PROMPT_ATTACK 表示 prompt attack
    """

    rows = []

    with open(input_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        # 檢查必要欄位
        if "jailbreak_query" not in reader.fieldnames:
            raise ValueError(
                "CSV 缺少必要欄位: jailbreak_query"
            )

        for row in reader:
            text = row["jailbreak_query"]

            # 跳過空資料
            if text is None or not text.strip():
                continue

            rows.append({
                "text": text.strip(),
                "label": "PROMPT_ATTACK"
            })

    # 建立輸出資料夾
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 寫入新的 CSV
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "label"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("完成！")
    print(f"輸入檔案：{input_file}")
    print(f"輸出檔案：{output_file}")
    print(f"提取資料：{len(rows)} 筆")
    print(f"label：PROMPT_ATTACK")


if __name__ == "__main__":

    # 直接在這裡指定檔案名稱
    input_file = "../JailBreakV_28K.csv"
    output_file = "extracted_prompt_attack.csv"

    extract_prompt_attacks(
        input_file,
        output_file
    )
