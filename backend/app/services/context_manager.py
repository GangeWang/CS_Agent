from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from ..config import settings
from .streamer import request_stream_sync

logger = logging.getLogger(__name__)

OLLAMA_GPT_OSS_20B_CONTEXT_TOKENS = 128 * 1024

SUMMARY_ROLE = "system"
SUMMARY_PREFIX = "[CONVERSATION_SUMMARY]"
SUMMARY_SUFFIX = "[/CONVERSATION_SUMMARY]"


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Lightweight token estimate used before calling the LLM service.

    CJK text often maps closer to 1 char/token, while English text is commonly
    ~4 chars/token. The conservative mixed estimate below prevents oversized
    prompts without requiring a tokenizer dependency for every model.
    """
    total = 0
    for msg in messages:
        content = str(msg.get("content", ""))
        role = str(msg.get("role", ""))
        cjk_chars = sum(1 for ch in content if "\u4e00" <= ch <= "\u9fff")
        non_cjk_chars = max(0, len(content) - cjk_chars)
        total += 4 + len(role) + cjk_chars + max(1, non_cjk_chars // 4)
    return total


def _configured_context_max_tokens() -> int:
    # Ollama lists gpt-oss:20b with a 128K context window. Use 128 * 1024
    # tokens as the project default while still allowing deployments to lower it
    # through LLAMA_CONTEXT_MAX_TOKENS when VRAM/RAM is constrained.
    return max(256, int(getattr(settings, "llama_context_max_tokens", OLLAMA_GPT_OSS_20B_CONTEXT_TOKENS)))


def _context_budget_tokens() -> int:
    max_context = _configured_context_max_tokens()
    reserved = max(0, int(getattr(settings, "llama_context_reserved_output_tokens", 4096)))
    return max(128, max_context - reserved)


def _threshold_tokens() -> int:
    ratio = float(getattr(settings, "context_compress_threshold_ratio", 0.8))
    ratio = min(0.95, max(0.1, ratio))
    return max(128, int(_context_budget_tokens() * ratio))


def _is_summary_message(message: Dict[str, str]) -> bool:
    return message.get("role") == SUMMARY_ROLE and message.get("content", "").startswith(SUMMARY_PREFIX)


def _split_summary_and_dialogue(history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    summaries: List[Dict[str, str]] = []
    dialogue: List[Dict[str, str]] = []
    for item in history:
        if _is_summary_message(item):
            summaries.append(item)
        else:
            dialogue.append(item)
    return summaries, dialogue


def _format_messages(messages: List[Dict[str, str]]) -> str:
    role_names = {
        "system": "系統摘要",
        "user": "使用者",
        "assistant": "客服助手",
    }
    lines: List[str] = []
    for item in messages:
        content = item.get("content", "").strip()
        if not content:
            continue
        role = role_names.get(item.get("role", ""), item.get("role", "訊息"))
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def _summarize_context_sync(messages: List[Dict[str, str]], model: str | None) -> str | None:
    if not messages:
        return None

    target_chars = max(300, int(getattr(settings, "context_summary_target_chars", 1200)))
    prompt = (
        "你正在為客服對話做上下文壓縮。請只保留後續回覆必需資訊，"
        "不要新增未出現在原文的內容，並使用繁體中文。\n"
        "摘要需包含：\n"
        "- 使用者需求、已確認條件、限制與偏好\n"
        "- 已提供的答案、承諾、流程、數字或政策重點\n"
        "- 尚未解決的問題與下一步\n"
        f"請控制在約 {target_chars} 字以內。\n\n"
        f"要壓縮的對話：\n{_format_messages(messages)}"
    )

    chunks: List[str] = []
    errors: List[str] = []

    def on_chunk(chunk: dict) -> None:
        if chunk.get("type") == "delta":
            chunks.append(chunk.get("text", ""))
        elif chunk.get("type") == "error":
            errors.append(chunk.get("error", ""))

    request_stream_sync(prompt, model, on_chunk, None)
    summary = "".join(chunks).strip()
    if summary:
        logger.info("Context compression summary generated:\n%s", summary)
        print(f"[context_summary]\n{summary}", flush=True)
        return f"{SUMMARY_PREFIX}\n以下是較早對話的壓縮摘要，請視為歷史上下文依據：\n{summary}\n{SUMMARY_SUFFIX}"

    if errors:
        logger.warning("Context compression failed: %s", errors[-1])
    else:
        logger.warning("Context compression returned empty summary")
    return None


def _window_history(history: List[Dict[str, str]], fixed_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    budget = _context_budget_tokens()
    keep_recent = max(0, int(getattr(settings, "context_window_keep_recent_messages", 8)))
    selected = list(history)

    while selected and estimate_tokens(fixed_messages + selected) > budget:
        if len(selected) <= keep_recent:
            break
        selected.pop(0)

    while selected and estimate_tokens(fixed_messages + selected) > budget:
        selected.pop(0)

    return selected


def _compress_history(
    history: List[Dict[str, str]],
    fixed_messages: List[Dict[str, str]],
    model: str | None,
) -> List[Dict[str, str]]:
    if estimate_tokens(fixed_messages + history) <= _threshold_tokens():
        return history

    summaries, dialogue = _split_summary_and_dialogue(history)
    keep_recent = max(2, int(getattr(settings, "context_compress_keep_recent_messages", 8)))
    older = dialogue[:-keep_recent] if len(dialogue) > keep_recent else []
    recent = dialogue[-keep_recent:] if len(dialogue) > keep_recent else dialogue

    if not older and summaries:
        return _window_history(summaries + recent, fixed_messages)

    summary_input = summaries + older
    summary = _summarize_context_sync(summary_input, model)
    if not summary:
        logger.warning("Falling back to context window because compression failed")
        return _window_history(history, fixed_messages)

    compressed = [{"role": SUMMARY_ROLE, "content": summary}] + recent
    if estimate_tokens(fixed_messages + compressed) > _context_budget_tokens():
        compressed = _window_history(compressed, fixed_messages)
    return compressed


def manage_context(
    history: List[Dict[str, str]],
    system_messages: List[Dict[str, str]],
    current_user_msg: str,
    model: str | None,
    strategy: str | None = None,
) -> tuple[List[Dict[str, str]], dict]:
    """
    Return a context-safe history for the next LLM request.

    Strategies:
    - compress: summarize older context with the LLM, keep recent turns verbatim.
    - window: drop oldest messages until the prompt fits the budget.
    """
    selected_strategy = (strategy or getattr(settings, "context_strategy", "compress") or "compress").lower()
    if selected_strategy not in {"compress", "window"}:
        selected_strategy = "compress"

    fixed_messages = system_messages + [{"role": "user", "content": current_user_msg}]
    before_tokens = estimate_tokens(fixed_messages + history)

    if before_tokens <= _threshold_tokens():
        return list(history), {
            "strategy": selected_strategy,
            "applied": False,
            "before_tokens": before_tokens,
            "after_tokens": before_tokens,
            "dropped_messages": 0,
        }

    if selected_strategy == "window":
        managed = _window_history(list(history), fixed_messages)
    else:
        managed = _compress_history(list(history), fixed_messages, model)

    after_tokens = estimate_tokens(fixed_messages + managed)
    return managed, {
        "strategy": selected_strategy,
        "applied": True,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "dropped_messages": max(0, len(history) - len(managed)),
    }
