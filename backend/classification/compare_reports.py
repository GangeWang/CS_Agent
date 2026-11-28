# -*- coding: utf-8 -*-
"""把 reports/*.json 並排成一張表，比較六種方法（與上線流程）在同一份 test / hard_eval / blind_eval 上的表現。

決定上線模型時看 blind_eval：hard_eval 已經被拿來補樣本，分數偏樂觀。

    python backend/classification/compare_reports.py

OvR 的分數是「4 個二元模型以上線決策規則組合後」的 4 類結果，可與 4選1 直接比較；
uncertain 欄是 OvR 沒有任何類別過門檻的比例（上線會再交給語意階段）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ORDER = ["ml_multiclass", "ml_ovr", "bilstm_multiclass", "bilstm_ovr",
         "transformer_multiclass", "transformer_ovr", "pipeline"]


def main() -> None:
    rows = []
    found = {p.stem for p in C.REPORT_DIR.glob("*.json")} if C.REPORT_DIR.exists() else set()
    # 依 ORDER 排序；換預訓練模型的報告（如 transformer_ovr__ckiplab-bert-base-chinese）排在對應方法後面
    names = [n for base in ORDER for n in sorted(found) if n == base or n.startswith(base + "__")]
    for name in names:
        p = C.REPORT_DIR / f"{name}.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        cells = [name]
        for split in ("test", "hard_eval", "blind_eval"):
            m = r.get(split)
            cells += ([f"{m['macro_f1']:.4f}", f"{m['accuracy']:.4f}", f"{m['uncertain_rate']:.2%}"]
                      if m else ["-", "-", "-"])
        rows.append(cells)
    if not rows:
        print(f"no reports under {C.REPORT_DIR}; run the training scripts first")
        return
    header = ["method", "test macroF1", "test acc", "test unc", "hard macroF1", "hard acc", "hard unc",
              "blind macroF1", "blind acc", "blind unc"]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for line in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths)))


if __name__ == "__main__":
    main()
