from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]

# 依你的實際路徑調整
MODEL_DIR = Path("./backend_ml_ovr_transformer_models_clean")  # 注意你原本「semantic_models」是全形底線，建議改半形 _
CONFIG_PATH = MODEL_DIR / "ovr_config.json"


def load_all():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH.resolve()}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = cfg["thresholds"]
    max_len = int(cfg.get("max_len", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    tokenizers = {}

    for lb in LABELS:
        subdir = MODEL_DIR / lb.lower()
        if not subdir.exists():
            raise FileNotFoundError(f"Model folder missing: {subdir.resolve()}")

        tokenizer = AutoTokenizer.from_pretrained(str(subdir))
        model = AutoModelForSequenceClassification.from_pretrained(str(subdir))
        model.to(device)
        model.eval()

        tokenizers[lb] = tokenizer
        models[lb] = model

    return models, tokenizers, thresholds, max_len, device


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_one_label(model, tokenizer, text: str, max_len: int, device) -> float:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits  # shape: [1,2] (binary)

    # 二元分類，取正類(1)機率
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return float(probs[1])


def predict_ovr(models, tokenizers, thresholds, text: str, max_len: int, device):
    probs = {}
    pass_flags = {}

    for lb in LABELS:
        prob_pos = predict_one_label(models[lb], tokenizers[lb], text, max_len, device)
        probs[lb] = prob_pos
        pass_flags[lb] = prob_pos >= float(thresholds[lb])

    # 1) 有通過 threshold 的類別，取 prob 最高者
    passed = [lb for lb in LABELS if pass_flags[lb]]
    if passed:
        best = max(passed, key=lambda x: probs[x])
        return best, probs, pass_flags

    # 2) 都沒過 -> UNCERTAIN
    return "UNCERTAIN", probs, pass_flags


def main():
    models, tokenizers, thresholds, max_len, device = load_all()

    print("=== Guardrail Transformer OvR Interactive Tester ===")
    print(f"Model dir: {MODEL_DIR.resolve()}")
    print(f"Device   : {device}")
    print(f"Max len  : {max_len}")
    print("Thresholds:")
    for lb in LABELS:
        print(f"  - {lb:14s}: {float(thresholds[lb]):.2f}")
    print("輸入 q / quit / exit 離開")

    while True:
        text = input("\n請輸入測試文字 > ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            print("bye.")
            break
        if not text:
            print("（空字串略過）")
            continue

        pred, probs, flags = predict_ovr(models, tokenizers, thresholds, text, max_len, device)

        print("-" * 72)
        print(f"Input      : {text}")
        print(f"Prediction : {pred}")
        print("Scores (is_label probability):")
        for lb in LABELS:
            mark = "PASS" if flags[lb] else "----"
            print(f"  - {lb:14s}: {probs[lb]:.4f}  (th={float(thresholds[lb]):.2f}) [{mark}]")
        if pred == "UNCERTAIN":
            print("Note       : 建議走 safe_fallback 或 semantic/LLM 二判")
        print("-" * 72)


if __name__ == "__main__":
    main()