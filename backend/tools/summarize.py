"""Conversation summarization tool."""
from __future__ import annotations

from typing import Dict, List

from backend.orchestrator.context import build_history_for_summary
from backend.tools.llm import request_stream_sync


def summarize_conversation_sync(history: List[Dict[str, str]], model: str | None) -> str:
    if not history:
        return "本次對話沒有可摘要的內容。"

    dialogue = build_history_for_summary(history)
    summary_prompt = (
        "請用繁體中文整理以下客服對話摘要，格式需包含：\n"
        "1. 問題重點\n"
        "2. 已提供的協助\n"
        "3. 後續建議（若無則寫無）\n\n"
        f"對話內容：\n{dialogue}"
    )

    chunks: List[str] = []
    error_text: List[str] = []

    def on_chunk(chunk: dict) -> None:
        if chunk.get("type") == "delta":
            chunks.append(chunk.get("text", ""))
        if chunk.get("type") == "error":
            error_text.append(chunk.get("error", ""))

    request_stream_sync(summary_prompt, model, on_chunk, None)

    summary = "".join(chunks).strip()
    if summary:
        return summary
    if error_text:
        return f"摘要產生失敗：{error_text[-1]}"
    return "摘要產生失敗，請稍後再試。"
