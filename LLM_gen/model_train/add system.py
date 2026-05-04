import json
from datasets import load_dataset

# 載入原始數據集
dataset = load_dataset("json", data_files="/home/gange/LLM_gen/sharegpt_data.json", split="train")

# 新的系統提示
new_system_prompt = "你是一個三星電子的客服，你需要用禮貌的語氣和客人解釋問題"

# 定義修改函數
def modify_system_message(example):
    messages = example["messages"]

    # 如果沒有 system role，則加入
    if messages[0]["role"] != "system":
        messages = [
            {"role": "system", "content": new_system_prompt}
        ] + messages

    return {"messages": messages}

# 應用修改
modified_dataset = dataset.map(modify_system_message)

# 保存修改後的數據集
modified_dataset.to_json(
    "/home/gange/LLM_gen/modified_sharegpt_data.json",
    force_ascii=False
)

print("完成加入 system message")