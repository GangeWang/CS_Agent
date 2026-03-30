from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import joblib

LABELS = ["NORMAL", "ABUSIVE", "PROMPT_ATTACK", "SPAM"]
MODEL_DIR = Path("../backend_ml_ovr_models")
CONFIG_PATH = MODEL_DIR / "ovr_config.json"


def load_all():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH.resolve()}")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = cfg["thresholds"]

    models = {}
    for lb in LABELS:
        p = MODEL_DIR / f"{lb.lower()}_bin.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Model missing: {p.resolve()}")
        models[lb] = joblib.load(p)

    return models, thresholds


def predict_ovr(models, thresholds, text: str):
    probs = {}
    pass_flags = {}

    for lb in LABELS:
        prob_pos = float(models[lb].predict_proba([text])[0, 1])  # 是該類的機率
        probs[lb] = prob_pos
        pass_flags[lb] = prob_pos >= float(thresholds[lb])

    # 決策：
    # 1) 有通過 threshold 的類別，取 prob 最高者
    passed = [lb for lb in LABELS if pass_flags[lb]]
    if passed:
        best = max(passed, key=lambda x: probs[x])
        return best, probs, pass_flags

    # 2) 都沒過 -> UNCERTAIN
    return "UNCERTAIN", probs, pass_flags


def main():
    models, thresholds = load_all()

    print("=== Guardrail OvR Interactive Tester ===")
    print(f"Model dir: {MODEL_DIR}")
    print("Thresholds:")
    for lb in LABELS:
        print(f"  - {lb:14s}: {thresholds[lb]}")
    print("輸入 q / quit / exit 離開")

    while True:
        text = input("\n請輸入測試文字 > ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            print("bye.")
            break
        if not text:
            print("（空字串略過）")
            continue

        pred, probs, flags = predict_ovr(models, thresholds, text)

        print("-" * 72)
        print(f"Input      : {text}")
        print(f"Prediction : {pred}")
        print("Scores (is_label probability):")
        for lb in LABELS:
            mark = "PASS" if flags[lb] else "----"
            print(f"  - {lb:14s}: {probs[lb]:.4f}  (th={thresholds[lb]:.2f}) [{mark}]")
        if pred == "UNCERTAIN":
            print("Note       : 建議走 safe_fallback 或 semantic/LLM 二判")
        print("-" * 72)


if __name__ == "__main__":
    main()