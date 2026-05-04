from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# ===== 改這三個 =====
BASE_MODEL = "unsloth/gpt-oss-20b"          # e.g. "meta-llama/Llama-3-8B"
LORA_PATH  = "lora_model"  # e.g. "./lora_model"
OUT_DIR    = "model_gguf"  # 輸出資料夾
# ====================

# 1) 載入 base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=1024,
    dtype=None,          # 自動
    load_in_4bit=True,  # 匯出前建議不要4bit載入
)

# 2) 載入 LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 這裡值其實不重要，真正權重會從 adapter 載入
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 如果你是已訓練好的 LoRA，通常直接這樣載入 adapter 比較保險：
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA_PATH)

# 3) 先存 LoRA（可選）
#model.save_pretrained("lora_model")
#tokenizer.save_pretrained("lora_model")

# 4) 直接輸出 GGUF (q4_k_m)
model.save_pretrained_gguf(
    OUT_DIR,
    tokenizer,
)
print(f"Done! GGUF exported to: {OUT_DIR}")