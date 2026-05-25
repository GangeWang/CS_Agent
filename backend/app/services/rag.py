"""Local RAG lookup service (Chinese-friendly, supports backend/app/data/rag_db.json)."""
from __future__ import annotations

import json
import logging
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


def _ngrams(text: str, n: int = 2) -> set[str]:
    t = _normalize(text).replace(" ", "")
    if not t:
        return set()
    if len(t) < n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


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
    score: int
    matched_question: str


def retrieve_rag_hit(user_query: str, *, min_score: int = 3) -> RagHit | None:
    """
    1) Exact match against question/aliases (normalized)
    2) Fallback: 2-gram overlap scoring (Chinese friendly)

    min_score: 2-gram overlap threshold. Default 3.
    """
    query_norm = _normalize(user_query)
    if not query_norm:
        return None

    records = _get_records_cached()
    if not records:
        return None

    # 1) exact match
    for rec in records:
        question_raw = rec.get("question", "")
        question_norm = _normalize(str(question_raw))
        aliases = rec.get("aliases", [])
        alias_norms = [_normalize(str(a)) for a in aliases if isinstance(a, str)]

        if query_norm == question_norm or query_norm in alias_norms:
            ans = rec.get("answer")
            if isinstance(ans, str) and ans.strip():
                return RagHit(answer=ans.strip(), score=9999, matched_question=str(question_raw))

    # 2) 2-gram overlap
    query_tokens = _ngrams(query_norm, 2)

    best_score = 0
    best_answer: Optional[str] = None
    best_question: str = ""

    for rec in records:
        question_raw = str(rec.get("question", ""))
        q_tokens = _ngrams(question_raw, 2)

        aliases = rec.get("aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str):
                    q_tokens |= _ngrams(alias, 2)

        if not q_tokens:
            continue

        score = len(query_tokens & q_tokens)
        if score > best_score:
            ans = rec.get("answer")
            if isinstance(ans, str) and ans.strip():
                best_score = score
                best_answer = ans.strip()
                best_question = question_raw

    if best_answer and best_score >= min_score:
        return RagHit(answer=best_answer, score=best_score, matched_question=best_question)

    return None


def retrieve_rag_answer(user_query: str) -> str | None:
    """Backward-compatible API."""
    hit = retrieve_rag_hit(user_query)
    return hit.answer if hit else None