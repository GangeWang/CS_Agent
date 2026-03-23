from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import os

# -----------------------------
# 模型設定
# -----------------------------
max_seq_length = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# -----------------------------
# 資料集
# -----------------------------
dataset = load_dataset(
    "json",
    data_files="/home/gange/CS_Agent/sharegpt.jsonl",
    split="train",
)

system_prompt = "你是一個專業的繁體中文客服，需要有耐心並使用繁體中文回答客人的問題"

def normalize_to_messages(example):
    if "messages" in example and example["messages"] is not None:
        msgs = []
        for m in example["messages"]:
            role = m.get("role")
            content = m.get("content", "")
            if role in ["system", "user", "assistant"] and content is not None:
                msgs.append({"role": role, "content": str(content)})
        if len(msgs) > 0:
            return msgs

    if "conversations" in example and example["conversations"] is not None:
        msgs = []
        for t in example["conversations"]:
            role = t.get("from", "")
            value = t.get("value", "")
            if role in ["human", "user"]:
                msgs.append({"role": "user", "content": str(value)})
            elif role in ["gpt", "assistant"]:
                msgs.append({"role": "assistant", "content": str(value)})
            elif role == "system":
                msgs.append({"role": "system", "content": str(value)})
        if len(msgs) > 0:
            return msgs

    return []

def formatting_func(examples):
    texts = []
    batch_size = len(next(iter(examples.values()))) if len(examples) > 0 else 0

    for i in range(batch_size):
        ex = {k: examples[k][i] for k in examples.keys()}
        msgs = normalize_to_messages(ex)

        if len(msgs) == 0:
            texts.append("")
            continue

        if not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": system_prompt}] + msgs

        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return {"text": texts}

dataset = dataset.map(
    formatting_func,
    batched=True,
    remove_columns=dataset.column_names,
)

dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 0)

print("dataset size:", len(dataset))
print(dataset[0]["text"][:500])

# -----------------------------
# Trainer 設定（支援 checkpoint）
# -----------------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        max_steps=10000,
        learning_rate=1e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
        max_grad_norm=1.0,
        # ---------- checkpoint 設定 ----------
        save_strategy="steps",      # 依步數存
        save_steps=200,             # 每 200 steps 存一次
        save_total_limit=5,         # 最多保留 5 個 checkpoint
    ),
)

# -----------------------------
# 訓練
# -----------------------------
# 如果有舊 checkpoint，從最後一次接續
latest_checkpoint = None
if os.path.exists(output_dir):
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if len(ckpts) > 0:
        ckpts.sort(key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = os.path.join(output_dir, ckpts[-1])
        print(f"Resume training from checkpoint: {latest_checkpoint}")

trainer.train(resume_from_checkpoint=latest_checkpoint)

# -----------------------------
# 推理測試
# -----------------------------
print("\n=== 測試 ===")
test_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "高雄建國服務中心的營業時間是幾點"},
]

prompt = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# -----------------------------
# 儲存模型
# -----------------------------
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")