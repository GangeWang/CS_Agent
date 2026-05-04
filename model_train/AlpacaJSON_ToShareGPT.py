import json


def alpaca_to_sharegpt(alpaca_path, output_path):
    with open(alpaca_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for entry in data:
        messages = []
        # 如果有 instruction
        if entry.get("instruction"):
            messages.append({"role": "user", "content": entry["instruction"]})
        # 如果有 input
        if entry.get("input"):
            messages.append({"role": "user", "content": entry["input"]})
        # output
        messages.append({"role": "assistant", "content": entry["output"]})

        new_data.append({"messages": messages})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


alpaca_to_sharegpt("data.json", "sharegpt_data.json")
