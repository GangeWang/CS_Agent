import json

input_file = "/model_train/combine_shuffle_train.jsonl"  # 你的原始檔，每行一個 JSON
output_file = "sharegpt.jsonl"  # 轉換後檔案

def convert_record(x):
    prompt = (x.get("prompt") or "").strip()
    completion = (x.get("completion") or "").strip()

    if not prompt or not completion:
        return None

    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": completion}
        ]
    }

count_in, count_out = 0, 0

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        count_in += 1
        obj = json.loads(line)
        new_obj = convert_record(obj)
        if new_obj is None:
            continue
        fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
        count_out += 1

print(f"done. input={count_in}, output={count_out}")