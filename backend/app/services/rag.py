"""
本機 FAQ 檢索（backend/app/data/rag_db.json）。

比對方式：
1) 正規化後與 question / aliases 完全相同 → 直接命中（score 1.0）
2) 否則用「去掉客服常見贅詞後的中文雙字詞 + 英數單字」做 IDF 加權比對：
   score = 0.7 × FAQ 問題被涵蓋的比例 + 0.3 × 使用者訊息被涵蓋的比例
   兩邊至少要有 2 個共同詞（FAQ 問題本身只有 1～2 個詞時例外），score ≥ RAG_MIN_SCORE（預設 0.4）才算命中。

先前的做法是「原始雙字詞重疊數 ≥ 3」：「請問」「可以」「直接」這類到處都有的詞就能湊滿 3 個，
在 199 句 hard_eval 上有 52 句誤命中（包含 13 句提示詞攻擊），而且 prompt 還要求模型「必須以此為核心回答」。
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import settings

logger = logging.getLogger(__name__)


def _resolve_rag_db_path() -> Path:
    """
    Priority:
    1) settings.rag_db_path (if you add it)
    2) backend/app/data/rag_db.json (relative to this file)
    3) backend/app/data/rag_db.json (relative to CWD if running from backend/)
    """
    # 1) optional: allow settings override
    cfg = getattr(settings, "rag_db_path", None)
    if isinstance(cfg, str) and cfg.strip():
        return Path(cfg).expanduser().resolve()

    # 2) relative to this module: backend/app/services/rag.py -> parents[1] = backend/app
    p = Path(__file__).resolve().parents[1] / "data" / "rag_db.json"
    if p.exists():
        return p

    # 3) fallback: relative to current working directory (backend/)
    return (Path.cwd() / "app" / "data" / "rag_db.json").resolve()


_RAG_DB_PATH = _resolve_rag_db_path()


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


# 客服訊息裡到處都有、不代表主題的詞：先換成空白再切詞，避免「請問」「怎麼」這類詞撐高分數，
# 也避免「如何退貨」切出「何退」這種跨詞的雜訊（長的先換，「怎麼辦」不會只剩「辦」）
_STOP_PHRASES = sorted(
    {
        "請問", "想問", "可以", "能不能", "不能", "怎麼辦", "要怎麼", "怎麼", "為什麼", "什麼", "如何",
        "是否", "是不是", "有沒有", "沒有", "需要", "想要", "不是", "這個", "那個", "現在", "已經",
        "直接", "幫我", "我要", "我想", "我的", "你們", "我們", "一下", "一直", "因為", "所以",
        "還是", "還有", "就是", "的話", "多少", "處理", "大概", "了嗎", "嗎", "呢", "吧", "啊", "的", "了",
    },
    key=len,
    reverse=True,
)
_STOP_WORDS = {
    "the", "and", "to", "my", "is", "a", "an", "you", "your", "me", "it", "of", "for", "in", "on",
    "this", "that", "can", "be", "with", "i", "what", "how", "do", "does", "are", "was", "have",
    "has", "not", "no", "please",
}
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+\-]*")


def _tokens(text: str) -> set[str]:
    """中文取雙字詞、英數取單字（英文不再切成字母雙字組，否則 'th'、'in' 會到處命中）。"""
    t = (text or "").lower()
    for phrase in _STOP_PHRASES:
        t = t.replace(phrase, " ")
    out: set[str] = set()
    for run in _CJK_RUN_RE.findall(t):
        out.update(run[i : i + 2] for i in range(len(run) - 1))
    out.update(w for w in _WORD_RE.findall(t) if len(w) >= 2 and w not in _STOP_WORDS)
    return out


def _load_rag_records() -> list[dict[str, Any]]:
    if not _RAG_DB_PATH.exists():
        logger.warning("RAG DB not found: %s (cwd=%s)", _RAG_DB_PATH, Path.cwd())
        return []

    try:
        data = json.loads(_RAG_DB_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse RAG DB at %s: %s", _RAG_DB_PATH, exc)
        return []

    if not isinstance(data, list):
        logger.warning("RAG DB must be a JSON array(list), got %s", type(data).__name__)
        return []

    records = [item for item in data if isinstance(item, dict)]
    logger.info("Loaded RAG records: %d from %s", len(records), _RAG_DB_PATH)
    return records


# ---- mtime cache (auto reload when file changes) ----
_cached_mtime: float | None = None
_cached_records: list[dict[str, Any]] = []


def _get_records_cached() -> list[dict[str, Any]]:
    global _cached_mtime, _cached_records

    try:
        mtime = _RAG_DB_PATH.stat().st_mtime
    except FileNotFoundError:
        _cached_mtime = None
        _cached_records = []
        return _cached_records

    if _cached_mtime is None or mtime != _cached_mtime:
        _cached_records = _load_rag_records()
        _cached_mtime = mtime

    return _cached_records


@dataclass
class RagHit:
    answer: str
    score: float  # 0～1；完全相同為 1.0
    matched_question: str


@dataclass
class _Doc:
    question: str
    answer: str
    exact_keys: set[str]
    tokens: set[str]


# 索引跟著 _get_records_cached() 回傳的 list 物件重建（檔案 mtime 變了才會換新的 list）
_index_source: list[dict[str, Any]] | None = None
_index: tuple[list[_Doc], dict[str, float]] = ([], {})


def _get_index() -> tuple[list[_Doc], dict[str, float]]:
    global _index_source, _index
    records = _get_records_cached()
    if records is _index_source:
        return _index

    docs: list[_Doc] = []
    for rec in records:
        answer = rec.get("answer")
        if not (isinstance(answer, str) and answer.strip()):
            continue
        question = str(rec.get("question", ""))
        aliases = [a for a in (rec.get("aliases") or []) if isinstance(a, str)]
        tokens = _tokens(question)
        for alias in aliases:
            tokens |= _tokens(alias)
        exact_keys = {_normalize(question)} | {_normalize(a) for a in aliases}
        docs.append(_Doc(question, answer.strip(), {k for k in exact_keys if k}, tokens))

    df: dict[str, int] = {}
    for doc in docs:
        for tok in doc.tokens:
            df[tok] = df.get(tok, 0) + 1
    n = max(1, len(docs))
    idf = {tok: math.log(1 + n / count) for tok, count in df.items()}

    _index_source, _index = records, (docs, idf)
    return _index


def retrieve_rag_hit(user_query: str, *, min_score: float | None = None) -> RagHit | None:
    """回傳最相近的一筆 FAQ；分數未達 min_score（預設 settings.rag_min_score）時回 None。"""
    query_norm = _normalize(user_query)
    if not query_norm:
        return None

    docs, idf = _get_index()
    if not docs:
        return None

    for doc in docs:
        if query_norm in doc.exact_keys:
            return RagHit(answer=doc.answer, score=1.0, matched_question=doc.question)

    threshold = settings.rag_min_score if min_score is None else min_score
    query_tokens = _tokens(query_norm)
    if not query_tokens:
        return None
    # 使用者訊息裡 FAQ 沒出現過的詞，當成最稀有的詞計算權重
    unseen_weight = max(idf.values(), default=1.0)
    query_weight = sum(idf.get(tok, unseen_weight) for tok in query_tokens)

    best: RagHit | None = None
    for doc in docs:
        common = query_tokens & doc.tokens
        if not common or (len(common) < 2 and len(doc.tokens) > 2):
            continue
        common_weight = sum(idf[tok] for tok in common)
        doc_coverage = common_weight / sum(idf[tok] for tok in doc.tokens)
        query_coverage = common_weight / query_weight
        score = 0.7 * doc_coverage + 0.3 * query_coverage
        if best is None or score > best.score:
            best = RagHit(answer=doc.answer, score=round(score, 3), matched_question=doc.question)

    if best is not None and best.score >= threshold:
        return best
    return None
