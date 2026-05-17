from __future__ import annotations

from typing import Any, Dict, List, Callable

from .executor import execute_plan
from .memory import append_short_term, get_short_term
from .planner import plan_steps
from ..streamer import request_stream_sync


def _extract_latest_user_message(messages: object) -> str | None:
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


def _ensure_respond_step(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if any(step.get("action") == "respond" for step in plan):
        return plan
    return plan + [{"step": len(plan) + 1, "action": "respond"}]


def run_agent(
    messages: object,
    model: str | None = None,
    session_id: int | str | None = None,
    on_chunk: Callable[[dict], None] | None = None,
) -> Dict[str, Any]:
    user_msg = _extract_latest_user_message(messages)
    if not user_msg and session_id is not None:
        history = get_short_term(session_id)
        for item in reversed(history):
            if item.get("role") == "user" and item.get("content"):
                user_msg = item["content"]
                break

    if not user_msg:
        return {
            "plan": [],
            "tool_results": [],
            "final": "缺少使用者訊息，無法執行 Agent。",
        }

    plan = _ensure_respond_step(plan_steps(user_msg))
    tool_results = execute_plan(plan)

    # 工具結果 → system content
    tool_summary_lines = []
    for result in tool_results:
        if "output" in result:
            tool_summary_lines.append(f"- {result['tool']}: {result['output']}")
        else:
            tool_summary_lines.append(f"- {result['tool']}: ERROR {result.get('error')}")
    tool_summary = "\n".join(tool_summary_lines).strip()

    agent_system_base = (
        "你是客服助理，請根據工具結果回答使用者問題。"
        "請全程使用自然、有人味的繁體中文。"
    )
    merged_system = agent_system_base
    if tool_summary:
        merged_system += "\n\n工具結果：\n" + tool_summary

    history = get_short_term(session_id) if session_id is not None else []

    # 注意：guardrail system 會在 ws.py 注入
    augmented_history = [{"role": "system", "content": merged_system}] + history

    final_chunks: List[str] = []
    error_text: List[str] = []

    def _emit(chunk: dict) -> None:
        if chunk.get("type") == "delta":
            final_chunks.append(chunk.get("text", ""))
        if chunk.get("type") == "error":
            error_text.append(chunk.get("error", ""))

        if on_chunk:
            on_chunk(chunk)

    llm_user_msg = f"使用者問題：{user_msg}"

    request_stream_sync(
        llm_user_msg,
        model,
        _emit,
        augmented_history,
    )

    final = "".join(final_chunks).strip()
    if not final:
        if error_text:
            final = f"LLM 生成失敗：{error_text[-1]}"
        else:
            final = "LLM 生成失敗，請稍後再試。"

    if session_id is not None:
        append_short_term(session_id, "user", user_msg)
        append_short_term(session_id, "assistant", final)

    return {
        "plan": plan,
        "tool_results": tool_results,
        "final": final,
    }