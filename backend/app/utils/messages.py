"""前端 payload.messages 的共用解析（ws.py 與 agent runner 共用，原本兩邊各寫一份）。"""
from __future__ import annotations

from typing import Dict, List


def extract_latest_user_message(messages: object) -> str | None:
    """
    從前端 payload.messages 取最後一筆有效 user 訊息。

    支援：
    - list[dict]（標準格式）
    - str（降級相容，視為 user 文字）
    """
    if isinstance(messages, str):
        text = messages.strip()
        return text or None

    if not isinstance(messages, list):
        return None

    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def extract_history_messages(messages: object) -> List[Dict[str, str]]:
    """取出最後一筆 user 訊息「之前」的 system / user / assistant 訊息（最後一筆會另外當成本輪提問）。"""
    if not isinstance(messages, list):
        return []

    normalized: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip():
            normalized.append({"role": role, "content": content.strip()})

    for idx in range(len(normalized) - 1, -1, -1):
        if normalized[idx].get("role") == "user":
            return normalized[:idx]
    return normalized
