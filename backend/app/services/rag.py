"""Simple local RAG lookup service."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_RAG_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "rag_db.json"


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _load_rag_records() -> list[dict[str, Any]]:
    if not _RAG_DB_PATH.exists():
        return []
    try:
        data = json.loads(_RAG_DB_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse RAG DB at %s: %s", _RAG_DB_PATH, exc)
        return []

    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def retrieve_rag_answer(user_query: str) -> str | None:
    """
    Return the best matching canonical answer from local RAG DB.

    Matching strategy:
    1) Exact match by normalized query / aliases
    2) Fallback keyword overlap scoring
    """
    query_norm = _normalize(user_query)
    if not query_norm:
        return None

    records = _load_rag_records()
    if not records:
        return None

    # 1) exact query / alias match
    for rec in records:
        question = _normalize(str(rec.get("question", "")))
        aliases = rec.get("aliases", [])
        alias_norms = [_normalize(str(a)) for a in aliases if isinstance(a, str)]
        if query_norm and (query_norm == question or query_norm in alias_norms):
            ans = rec.get("answer")
            if isinstance(ans, str) and ans.strip():
                return ans.strip()

    # 2) keyword overlap
    query_tokens = set(query_norm.split())
    best_score = 0
    best_answer: str | None = None
    for rec in records:
        q_tokens = set(_normalize(str(rec.get("question", ""))).split())
        aliases = rec.get("aliases", [])
        for alias in aliases if isinstance(aliases, list) else []:
            if isinstance(alias, str):
                q_tokens |= set(_normalize(alias).split())
        if not q_tokens:
            continue
        score = len(query_tokens & q_tokens)
        if score > best_score:
            ans = rec.get("answer")
            if isinstance(ans, str) and ans.strip():
                best_score = score
                best_answer = ans.strip()

    # Require at least minimal overlap to avoid noisy hits
    return best_answer if best_score >= 2 else None

