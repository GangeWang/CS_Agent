from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List

from ...utils.messages import extract_history_messages, extract_latest_user_message
from .executor import execute_plan
from .memory import append_short_term, get_short_term
from .planner import plan_steps
from ..streamer import stream_chat

logger = logging.getLogger(__name__)


def _ensure_respond_step(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if any(step.get("action") == "respond" for step in plan):
        return plan
    return plan + [{"step": len(plan) + 1, "action": "respond"}]


async def run_agent_stream(
    messages: object,
    model: str | None = None,
    session_id: int | str | None = None,
) -> AsyncIterator[dict]:
    """
    Agent 模式：規劃 → 執行工具 → 把工具結果交給 LLM 串流回答。

    逐段 yield delta / done / error（與 streamer.stream_chat 相同格式），呼叫端（ws.py）負責轉送與逾時。
    plan 與工具結果只寫進後端 log，不送到前端。
    """
    user_msg = extract_latest_user_message(messages)
    if not user_msg and session_id is not None:
        for item in reversed(get_short_term(session_id)):
            if item.get("role") == "user" and item.get("content"):
                user_msg = item["content"]
                break

    if not user_msg:
        yield {"type": "error", "error": "缺少使用者訊息，無法執行 Agent。"}
        return

    plan = _ensure_respond_step(plan_steps(user_msg))
    # 工具目前都是 stub；放進執行緒池，日後換成真的 I/O 也不會阻塞 event loop
    tool_results = await asyncio.to_thread(execute_plan, plan)
    logger.info("agent plan=%s", plan)
    logger.info("agent tool_results=%s", tool_results)

    # 工具結果 → system content
    tool_summary_lines = []
    for result in tool_results:
        if "output" in result:
            tool_summary_lines.append(f"- {result['tool']}: {result['output']}")
        else:
            tool_summary_lines.append(f"- {result['tool']}: ERROR {result.get('error')}")
    tool_summary = "\n".join(tool_summary_lines).strip()

    merged_system = (
        "你是客服助理，請根據工具結果回答使用者問題。"
        "請全程使用自然、有人味的繁體中文。"
    )
    if tool_summary:
        merged_system += "\n\n工具結果：\n" + tool_summary

    if session_id is not None:
        history = get_short_term(session_id)
    else:
        history = extract_history_messages(messages)

    # 注意：
    # 1) context 管理（manage_context）由呼叫端（ws.py）統一執行並持久化到 session，
    #    這裡不再重做，否則同一輪會對 LLM 做兩次壓縮摘要。
    # 2) guardrail system 會在 ws.py 注入（存在傳入 messages 的 system role 中）。
    augmented_history = [{"role": "system", "content": merged_system}] + history

    final_chunks: List[str] = []
    async for chunk in stream_chat(f"使用者問題：{user_msg}", model, augmented_history):
        if chunk.get("type") == "delta":
            final_chunks.append(chunk.get("text", ""))
        elif chunk.get("type") == "error":
            # 錯誤細節只進 log
            logger.warning("Agent LLM generation failed: %s", chunk.get("error"))
        yield chunk

    final = "".join(final_chunks).strip()
    if session_id is not None and final:
        append_short_term(session_id, "user", user_msg)
        append_short_term(session_id, "assistant", final)
