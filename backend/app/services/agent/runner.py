from __future__ import annotations

from typing import Any, Dict, List

from .executor import execute_plan
from .memory import append_short_term, get_short_term
from .planner import plan_steps


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


def run_agent(messages: object, model: str | None = None, session_id: int | str | None = None) -> Dict[str, Any]:
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

    final = "以下是工具執行結果：\n"
    for result in tool_results:
        if "output" in result:
            final += f"- {result['tool']}: {result['output']}\n"
        else:
            final += f"- {result['tool']}: ERROR {result.get('error')}\n"
    final = final.strip()

    if session_id is not None:
        append_short_term(session_id, "user", user_msg)
        append_short_term(session_id, "assistant", final)

    return {
        "plan": plan,
        "tool_results": tool_results,
        "final": final,
    }
