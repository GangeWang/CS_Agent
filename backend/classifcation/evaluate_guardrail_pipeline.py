# -*- coding: utf-8 -*-
"""端到端評估「上線實際的 guardrail 流程」：RULE → ML(OvR transformer) → SEMANTIC → UNCERTAIN。

直接呼叫 backend/app/services/guardrail.py 的 classify_text，因此評估的是上線真正會跑的東西
（規則、門檻、決策邏輯、語意庫全部包含在內），而不是單一模型的分數。

評估集：
- hard_eval.csv（困難樣本，不進訓練）← 最重要
- 共用切分的 test（common.split_dataset）；注意若上線模型不是用 v2 + 本切分訓練的，test 可能與其訓練資料重疊

用法（於任意目錄執行，需 torch / transformers / sentence-transformers 與 LFS 模型檔）：
    python backend/classifcation/evaluate_guardrail_pipeline.py [--skip-test]
報告存 reports/pipeline.json（含各 stage 命中比例與錯誤樣本）。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # backend/，讓 app.services.guardrail 可 import
import common as C  # noqa: E402


def run(texts, y_true, title):
    from app.services.guardrail import classify_text

    outs = [classify_text(str(t)) for t in texts]
    if any(not o.get("available", True) for o in outs):
        print("[WARN] guardrail 模型不可用（降級為規則），結果不代表模型表現；請確認 LFS 模型檔已下載")
    y_pred = np.array([o["label"] for o in outs], dtype=object)
    res = C.evaluate_split(texts, y_true, y_pred, title)
    res["stage_counts"] = dict(Counter(o["stage"] for o in outs))
    for e, o in zip(res["errors"], [o for o, a, b in zip(outs, y_true, y_pred) if a != b]):
        e["stage"], e["reason"] = o.get("stage"), o.get("reason")
    print("stage counts:", res["stage_counts"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-test", action="store_true", help="只跑 hard_eval（較快）")
    args = ap.parse_args()

    results = {"method": "pipeline"}
    X_hard, y_hard = C.load_hard_eval()
    if X_hard is not None:
        results["hard_eval"] = run(X_hard, y_hard, "pipeline / hard_eval")
    if not args.skip_test:
        s = C.split_dataset(C.load_dataset())
        results["test"] = run(s.X_test, s.y_test, "pipeline / test")
    C.save_report("pipeline", results)


if __name__ == "__main__":
    main()
