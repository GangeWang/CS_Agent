from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("lora_model", use_fast=False)
print(tok.chat_template)