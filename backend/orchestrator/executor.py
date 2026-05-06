"""Main WebSocket chat orchestration flow."""
from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import settings
from backend.orchestrator.context import extract_latest_user_message
from backend.orchestrator.state import IDLE_TIMEOUT_SECONDS, IDLE_WARNING_SECONDS_BEFORE_END
from backend.tools.jsonsafe import json_dumps
from backend.tools.llm import request_stream_sync
from backend.tools.memory import (
    append_and_trim_history,
    clear_session,
    create_session,
    delete_session,
    get_history,
    set_history,
)
from backend.tools.safety import build_guardrail_instruction, classify_text
from backend.tools.summarize import summarize_conversation_sync

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_event_loop()
    session_id = id(websocket)
    create_session(session_id)
    last_dialogue_at = loop.time()
    idle_warning_sent = False
    last_model: str | None = None
    logger.info(f"WebSocket connection established: session_id={session_id}")

    async def end_conversation(reason: str, model: str | None = None) -> None:
        history: List[Dict[str, str]] = get_history(session_id)
        summary = await asyncio.to_thread(summarize_conversation_sync, history, model)
        logger.info(f"Conversation ended: session_id={session_id}, reason={reason}, summary={summary}")
        await websocket.send_text(json_dumps({
            "type": "conversation_summary",
            "reason": reason,
            "summary": "對話已關閉",
        }))
        await websocket.send_text(json_dumps({
            "type": "conversation_ended",
            "reason": reason,
        }))
        await websocket.close(code=1000, reason="conversation ended")

    try:
        while True:
            raw = await websocket.receive_text()

            if len(raw) > settings.max_message_size:
                await websocket.send_text(json_dumps({
                    "type": "error",
                    "error": f"訊息過大 (最大 {settings.max_message_size} bytes)",
                }))
                continue

            try:
                payload = json.loads(raw)
            except Exception as e:
                logger.warning(f"Invalid JSON from session {session_id}: {e}")
                await websocket.send_text(json_dumps({"type": "error", "error": "JSON 格式不正確"}))
                continue

            if payload.get("type") == "ping":
                idle_elapsed = loop.time() - last_dialogue_at
                if idle_elapsed > IDLE_TIMEOUT_SECONDS:
                    await end_conversation("idle_timeout", last_model)
                    break
                if (
                    not idle_warning_sent
                    and idle_elapsed >= (IDLE_TIMEOUT_SECONDS - IDLE_WARNING_SECONDS_BEFORE_END)
                    and idle_elapsed < IDLE_TIMEOUT_SECONDS
                ):
                    remaining_seconds = max(0, math.ceil(IDLE_TIMEOUT_SECONDS - idle_elapsed))
                    await websocket.send_text(json_dumps({
                        "type": "idle_warning",
                        "remaining_seconds": remaining_seconds,
                    }))
                    idle_warning_sent = True
                await websocket.send_text(json_dumps({"type": "pong"}))
                continue

            if payload.get("type") == "clear_history":
                clear_session(session_id)
                last_dialogue_at = loop.time()
                idle_warning_sent = False
                await websocket.send_text(json_dumps({"type": "history_cleared"}))
                continue

            if payload.get("type") == "end_conversation":
                await end_conversation("manual", last_model)
                break

            messages = payload.get("messages", [])
            user_msg = extract_latest_user_message(messages)
            if not user_msg:
                await websocket.send_text(json_dumps({"type": "error", "error": "缺少使用者訊息"}))
                continue
            last_dialogue_at = loop.time()
            idle_warning_sent = False

            guardrail_label = classify_text(user_msg).get("label", "NORMAL")
            guardrail_instruction = build_guardrail_instruction(guardrail_label)

            await websocket.send_text(json_dumps({"type": "guardrail", "label": guardrail_label}))

            model = payload.get("model")
            if isinstance(model, str) and model.strip():
                last_model = model
            history: List[Dict[str, str]] = get_history(session_id)
            set_history(session_id, history)

            # Keep request_stream_sync unchanged: inject the policy instruction into history.
            augmented_history = [{"role": "system", "content": guardrail_instruction}] + history.copy()

            q: asyncio.Queue = asyncio.Queue()
            assistant_response: List[str] = []

            def on_chunk(chunk: dict) -> None:
                try:
                    if chunk.get("type") == "delta":
                        assistant_response.append(chunk.get("text", ""))

                    def _put() -> None:
                        try:
                            q.put_nowait(chunk)
                        except asyncio.QueueFull:
                            logger.warning(f"Queue full for session {session_id}")

                    loop.call_soon_threadsafe(_put)
                except Exception as cb_err:
                    logger.exception(f"on_chunk failed for session {session_id}: {cb_err}")

            task = loop.run_in_executor(
                None,
                request_stream_sync,
                user_msg,
                model,
                on_chunk,
                augmented_history,
            )

            try:
                while True:
                    chunk = await asyncio.wait_for(q.get(), timeout=120)
                    await websocket.send_text(json_dumps(chunk))

                    if chunk.get("type") in ("done", "error"):
                        if chunk.get("type") == "done" and assistant_response:
                            append_and_trim_history(session_id, user_msg, "".join(assistant_response))
                        break

            except asyncio.TimeoutError:
                logger.error(f"Streaming timeout for session {session_id}")
                await websocket.send_text(json_dumps({"type": "error", "error": "模型回應逾時"}))
            except Exception as e:
                logger.exception(f"Error processing response for session {session_id}: {e}")
                await websocket.send_text(json_dumps({"type": "error", "error": "處理回應時發生錯誤"}))
            finally:
                try:
                    await task
                except Exception as e:
                    logger.exception(f"Executor task failed for session {session_id}: {e}")
                    await websocket.send_text(json_dumps({"type": "error", "error": "LLM 任務執行失敗"}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session_id={session_id}")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler for session {session_id}: {e}")
    finally:
        delete_session(session_id)
        logger.info(f"Session cleaned up: session_id={session_id}")
