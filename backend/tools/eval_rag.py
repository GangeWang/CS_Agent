"""
FAQ 檢索評估：換了 rag_db.json 或調整 RAG_MIN_SCORE 後跑一次，看命中率與誤命中。

跑法（在 backend/ 底下）：
    python tools/eval_rag.py                     # 用預設門檻（settings.rag_min_score）
    python tools/eval_rag.py --min-score 0.4     # 試別的門檻

兩組資料：
- tools/rag_eval_queries.csv：使用者換句話問 FAQ 裡的題目（應命中，且要命中對的那題）。
  換成自己的 FAQ 後，這份也要換成對應的問法。
- classifcation/data_set/hard_eval.csv：guardrail 的 199 句評估集，裡面的句子 FAQ 都沒有標準答案，
  命中就算誤命中（同主題但不同題目的會列出來，自行判斷）。
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.rag import retrieve_rag_hit  # noqa: E402

QUERIES = BACKEND / "tools" / "rag_eval_queries.csv"
HARD_EVAL = BACKEND / "classifcation" / "data_set" / "hard_eval.csv"


def _same(a: str, b: str) -> bool:
    return "".join(a.lower().split()) == "".join(b.lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(QUERIES.open(encoding="utf-8-sig")))
    correct, wrong, missed = 0, [], []
    for row in rows:
        hit = retrieve_rag_hit(row["query"], min_score=args.min_score)
        if hit is None:
            missed.append(row["query"])
        elif _same(hit.matched_question, row["expected_question"]):
            correct += 1
        else:
            wrong.append((row["query"], hit.matched_question.strip(), hit.score))
    print(f"應命中：{correct}/{len(rows)} 命中正確，{len(wrong)} 命中錯題，{len(missed)} 漏掉")
    for q, m, s in wrong:
        print(f"  錯題  {q} → {m} ({s})")
    for q in missed:
        print(f"  漏掉  {q}")

    if HARD_EVAL.exists():
        total, hits, examples = collections.Counter(), collections.Counter(), []
        for row in csv.DictReader(HARD_EVAL.open(encoding="utf-8-sig")):
            total[row["label"]] += 1
            hit = retrieve_rag_hit(row["text"], min_score=args.min_score)
            if hit:
                hits[row["label"]] += 1
                examples.append((row["label"], row["text"][:40], hit.matched_question.strip()[:30], hit.score))
        summary = "，".join(f"{lb} {hits[lb]}/{n}" for lb, n in total.items())
        print(f"hard_eval 誤命中：{sum(hits.values())}/{sum(total.values())}（{summary}）")
        for ex in examples:
            print("  ", ex)


if __name__ == "__main__":
    main()
