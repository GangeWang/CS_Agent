"""Conversation memory helpers for WebSocket chat sessions."""
from __future__ import annotations

from typing import Dict, List

from cachetools import TTLCache

from backend.config import settings

conversation_sessions: TTLCache = TTLCache(maxsize=1000, ttl=3600)


def create_session(session_id: int) -> None:
    conversation_sessions[session_id] = []


def get_history(session_id: int) -> List[Dict[str, str]]:
    return conversation_sessions.get(session_id, [])


def set_history(session_id: int, history: List[Dict[str, str]]) -> None:
    conversation_sessions[session_id] = history


def append_and_trim_history(session_id: int, user_msg: str, assistant_msg: str) -> None:
    history = get_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    if len(history) > settings.history_max_length:
        history = history[-settings.history_max_length:]
    set_history(session_id, history)


def clear_session(session_id: int) -> None:
    set_history(session_id, [])


def delete_session(session_id: int) -> None:
    conversation_sessions.pop(session_id, None)
