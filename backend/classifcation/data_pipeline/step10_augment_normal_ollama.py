# -*- coding: utf-8 -*-
"""Step10（選用）：以本機 ollama 依人工種子擴增「長句、真實語氣」的 NORMAL 樣本。

背景：清理後 NORMAL 幾乎全是 ML_data 的模板短句（中位數 11 字），人工種子只有幾十筆長句。
模型因此容易學到「長訊息 ≠ NORMAL」的長度捷徑。這支腳本以 curated_seed_data.NORMAL 為範例，
請本地模型產生不同情境的真實客服提問，輸出後務必人工抽查再併入訓練集。

用法（於 backend/classifcation/ 執行，需本機 ollama）：
    python data_pipeline/step10_augment_normal_ollama.py --n 1500
    # 抽查 data_set/normal_augmented.csv 後，把通過的列加入 curated_seed_data.NORMAL 或直接併入 v2
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curated_seed_data as seeds  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OUT = HERE.parent / "data_set" / "normal_augmented.csv"

SCENARIOS = [
    "訂單與物流", "退換貨與保固", "帳號登入與密碼", "付款與發票", "會員與點數", "產品使用問題",
    "門市與營業時間", "帳單與費率", "收到可疑簡訊想確認是否詐騙",
    "替家人代為詢問", "英文或中英夾雜的提問", "含有網址或訂單編號的提問", "含有『立即、投資、加入、系統、規則』等字眼的正常提問",
]

PROMPT = """你在協助建立客服分類器的訓練資料。請寫 5 則「一般顧客」傳給線上客服的訊息，情境：{scenario}。
要求：繁體中文（台灣用語），口語自然，長度 15～120 字不等，彼此情境不同；語氣平穩有禮，不得帶有不滿、焦急、失望、憤怒等情緒（有情緒的訊息屬於另一類），也不得含詐騙廣告或要求 AI 違反規則的內容。
參考語氣（不要照抄）：
{examples}
只輸出 5 行訊息本身，每行一則，不要編號、不要其他說明。"""


def gen(model: str, scenario: str) -> list[str]:
    ex = "\n".join(random.sample(seeds.NORMAL, 4))
    r = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": PROMPT.format(scenario=scenario, examples=ex),
        "stream": False, "options": {"temperature": 0.9},
    }, timeout=180)
    r.raise_for_status()
    lines = [x.strip(" -・•\t") for x in (r.json().get("response") or "").splitlines()]
    return [x for x in lines if 8 <= len(x) <= 160]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="huihui_ai/gpt-oss-abliterated:20b")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    rows: list[str] = []
    while len(rows) < a.n:
        try:
            rows += gen(a.model, random.choice(SCENARIOS))
        except Exception as e:  # 單次失敗就跳過，不保留任何半成品
            print("skip:", e)
        print(f"{len(rows)}/{a.n}", end="\r")
    pd.DataFrame({"text": rows[: a.n], "label": "NORMAL"}).drop_duplicates("text").to_csv(
        OUT, index=False, encoding="utf-8-sig")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
