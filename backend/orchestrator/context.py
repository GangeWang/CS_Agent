"""Context extraction and formatting helpers for chat orchestration."""
from __future__ import annotations

from typing import Dict, List


def extract_latest_user_message(messages: object) -> str | None:
    """Return the last valid user message from frontend payload.messages."""
    if isinstance(messages, str):
        text = messages.strip()
        return text if text else None

    if not isinstance(messages, list):
        return None

    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def build_history_for_summary(history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "user":
            lines.append(f"使用者：{content}")
        elif role == "assistant":
            lines.append(f"客服助手：{content}")
    return "\n".join(lines)
