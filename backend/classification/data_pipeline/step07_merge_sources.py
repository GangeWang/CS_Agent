# -*- coding: utf-8 -*-
"""Step07：合併四個翻譯/整理後的來源為全量資料集。

還原 2026-08-21 10:13 的手動合併步驟（事後補寫）。
歷史產物 merged_full_dataset.csv = 33,041 筆，組成已驗證：

    abusive_translated_zh.csv   9,433  (ABUSIVE)
    spam_translated_zh.csv        747  (spam → 後續步驟注意大小寫)
    prompt_attack_translated_zh.csv 4,172  (PROMPT_ATTACK)
    data_set/ML_data.csv       18,689  (7 月舊世代，含 NORMAL)
    ─────────────────────────────────────
    合計                       33,041  → 再經 step08 去重 → 21,567 筆訓練集

用法：
    python step07_merge_sources.py [輸出檔]
    # 預設輸出 merged_full_dataset.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

SOURCES = [
    ("abusive_translated_zh.csv", "ABUSIVE"),
    ("spam_translated_zh.csv", "SPAM"),
    ("prompt_attack_translated_zh.csv", "PROMPT_ATTACK"),
    ("data_set/ML_data.csv", None),  # label 以檔案現值為準（NORMAL 為主）
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", nargs="?", default="merged_full_dataset.csv")
    args = ap.parse_args()

    frames = []
    for path, normalize in SOURCES:
        if not Path(path).exists():
            print(f"[警告] 來源不存在，略過：{path}", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        if normalize and "label" in df.columns:
            df["label"] = df["label"].astype(str).str.upper().str.strip()
        frames.append(df)
        print(f"{path}: {len(df)} 筆", file=sys.stderr)

    merged = pd.concat(frames, ignore_index=True)
    # 基本清理：丟掉 text 空值
    merged = merged.dropna(subset=["text"])
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"合併完成：{len(merged)} 筆 → {args.output}")
    print("提醒：下一步用 step08 去重 → data_set/train_full_dataset.csv")


if __name__ == "__main__":
    main()
