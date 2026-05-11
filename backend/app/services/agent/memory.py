from __future__ import annotations

from typing import Dict, List

from cachetools import TTLCache


_SHORT_TERM_CACHE: TTLCache = TTLCache(maxsize=1000, ttl=3600)


def get_short_term(session_id: int | str) -> List[Dict[str, str]]:
    history = _SHORT_TERM_CACHE.get(session_id, [])
    return list(history)


def append_short_term(session_id: int | str, role: str, content: str) -> None:
    history: List[Dict[str, str]] = _SHORT_TERM_CACHE.get(session_id, [])
    history.append({"role": role, "content": content})
    _SHORT_TERM_CACHE[session_id] = history


def clear_short_term(session_id: int | str) -> None:
    _SHORT_TERM_CACHE.pop(session_id, None)


def get_long_term(session_id: int | str) -> List[Dict[str, str]]:
    # MVP stub：保留接口，後續可接向量庫
    return []


def write_long_term(session_id: int | str, payload: Dict[str, str]) -> None:
    # MVP stub：保留接口
    return None
