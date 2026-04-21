"""
WebSocket router for real-time chat with conversation memory.
Handles streaming responses from Ollama and maintains conversation history.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio
import logging
import math
from typing import Dict, List
from cachetools import TTLCache

from ..services.streamer import request_stream_sync
from ..services.guardrail import classify_text
from ..utils.jsonsafe import json_dumps
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

conversation_sessions: TTLCache = TTLCache(maxsize=1000, ttl=3600)
IDLE_TIMEOUT_SECONDS = 180
IDLE_WARNING_SECONDS_BEFORE_END = 60


def _append_and_trim_history(session_id: int, user_msg: str, assistant_msg: str) -> None:
    history: List[Dict[str, str]] = conversation_sessions.get(session_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    if len(history) > settings.history_max_length:
        history = history[-settings.history_max_length:]
    conversation_sessions[session_id] = history


def _build_guardrail_instruction(label: str) -> str:
    if label == "ABUSIVE":
        return (
            "你是客服助理。使用者情緒可能較激動，請先簡短同理與降溫，"
            "語氣保持禮貌、穩定、專業；接著再回答問題。避免說教、避免指責。"
        )
    if label in {"PROMPT_ATTACK", "SPAM"}:
        return (
            "你是客服助理。此請求可能涉及不當或無關內容。"
            "請婉拒不適當部分，不提供危險、違規或濫用指引；"
            "語氣禮貌簡潔，並引導使用者提出可協助的正當需求。"
        )
    return "你是客服助理。請直接、清楚、禮貌地回覆使用者問題。"


def _build_history_for_summary(history: List[Dict[str, str]]) -> str:
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


def _summarize_conversation_sync(history: List[Dict[str, str]], model: str | None) -> str:
    if not history:
        return "本次對話沒有可摘要的內容。"

    dialogue = _build_history_for_summary(history)
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

    request_stream_sync(
        summary_prompt,
        model,
        on_chunk,
        None,
    )

    summary = "".join(chunks).strip()
    if summary:
        return summary
    if error_text:
        return f"摘要產生失敗：{error_text[-1]}"
    return "摘要產生失敗，請稍後再試。"


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_event_loop()
    session_id = id(websocket)
    conversation_sessions[session_id] = []
    last_dialogue_at = loop.time()
    idle_warning_sent = False
    last_model: str | None = None
    logger.info(f"WebSocket connection established: session_id={session_id}")

    async def end_conversation(reason: str, model: str | None = None) -> None:
        history: List[Dict[str, str]] = conversation_sessions.get(session_id, [])
        summary = await asyncio.to_thread(_summarize_conversation_sync, history, model)
        await websocket.send_text(json_dumps({
            "type": "conversation_summary",
            "reason": reason,
            "summary": summary
        }))
        await websocket.send_text(json_dumps({
            "type": "conversation_ended",
            "reason": reason
        }))
        await websocket.close(code=1000, reason="conversation ended")

    try:
        while True:
            raw = await websocket.receive_text()

            if len(raw) > settings.max_message_size:
                await websocket.send_text(json_dumps({
                    "type": "error",
                    "error": f"訊息過大 (最大 {settings.max_message_size} bytes)"
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
                        "remaining_seconds": remaining_seconds
                    }))
                    idle_warning_sent = True
                await websocket.send_text(json_dumps({"type": "pong"}))
                continue

            if payload.get("type") == "clear_history":
                conversation_sessions[session_id] = []
                last_dialogue_at = loop.time()
                idle_warning_sent = False
                await websocket.send_text(json_dumps({"type": "history_cleared"}))
                continue

            if payload.get("type") == "end_conversation":
                await end_conversation("manual", last_model)
                break

            messages = payload.get("messages", [])
            user_msg = None
            if isinstance(messages, list):
                for m in reversed(messages):
                    if isinstance(m, dict) and m.get("role") == "user":
                        user_msg = m.get("content")
                        break
            if not user_msg:
                await websocket.send_text(json_dumps({"type": "error", "error": "缺少使用者訊息"}))
                continue
            last_dialogue_at = loop.time()
            idle_warning_sent = False

            guardrail_label = classify_text(user_msg).get("label", "NORMAL")
            guardrail_instruction = _build_guardrail_instruction(guardrail_label)

            await websocket.send_text(json_dumps({"type": "guardrail", "label": guardrail_label}))

            model = payload.get("model")
            if isinstance(model, str) and model.strip():
                last_model = model
            history: List[Dict[str, str]] = conversation_sessions[session_id]

            # 不改 request_stream_sync 簽名：把 system instruction 注入 history
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
                            _append_and_trim_history(session_id, user_msg, "".join(assistant_response))
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
        conversation_sessions.pop(session_id, None)
        logger.info(f"Session cleaned up: session_id={session_id}")
