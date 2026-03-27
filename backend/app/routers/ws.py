"""
WebSocket router for real-time chat with conversation memory.
Handles streaming responses from Ollama and maintains conversation history.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio
import logging
from typing import Dict, List
from cachetools import TTLCache

from ..services.streamer import request_stream_sync
from ..utils.jsonsafe import json_dumps
from ..config import settings

router = APIRouter()

# Configure logger
logger = logging.getLogger(__name__)

# Use TTL cache to prevent memory leaks:
# - key: session_id（每個 WebSocket 連線唯一識別）
# - value: 對話歷史 list[{"role","content"}]
# - ttl: 3600 秒（1 小時）後自動過期，避免長時間累積佔用記憶體
conversation_sessions: TTLCache = TTLCache(maxsize=1000, ttl=3600)


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for chat streaming with conversation memory.
    
    Features:
    - Streaming responses from Ollama
    - Conversation history management
    - Heartbeat support
    - Request size validation
    - Automatic session cleanup
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    # Generate unique session_id for this connection
    session_id = id(websocket)
    conversation_sessions[session_id] = []
    
    logger.info(f"WebSocket connection established: session_id={session_id}")

    try:
        while True:
            # 1) 接收前端送來的一段文字（JSON）
            raw = await websocket.receive_text()

            # Validate message size to prevent DoS attacks
            if len(raw) > settings.max_message_size:
                await websocket.send_text(json_dumps({
                    "type": "error",
                    "error": f"訊息過大 (最大 {settings.max_message_size} bytes)"
                }))
                logger.warning(f"Message too large: {len(raw)} bytes from session {session_id}")
                continue

            try:
                # 2) 解析 payload；格式錯誤時回傳 error 並等待下一筆
                payload = json.loads(raw)
            except Exception as e:
                await websocket.send_text(json_dumps({"type": "error", "error": "JSON 格式不正確"}))
                logger.warning(f"Invalid JSON from session {session_id}: {e}")
                continue

            # Handle heartbeat
            if payload.get("type") == "ping":
                # 前端心跳：立即回 pong，讓前端判斷連線活性
                await websocket.send_text(json_dumps({"type": "pong"}))
                continue

            # Handle clear history command
            if payload.get("type") == "clear_history":
                # 清空此連線會話歷史，不影響其他連線
                conversation_sessions[session_id] = []
                await websocket.send_text(json_dumps({"type": "history_cleared"}))
                logger.info(f"History cleared for session {session_id}")
                continue

            messages = payload.get("messages", [])
            # 只取本次 user 訊息（前端使用單輪送出；歷史由後端維護）
            user_msg = next((m.get("content") for m in messages if m.get("role") == "user"), None)
            if not user_msg:
                await websocket.send_text(json_dumps({"type": "error", "error": "缺少使用者訊息"}))
                continue

            model = payload.get("model")

            # Get current conversation history
            history: List[Dict[str, str]] = conversation_sessions[session_id]

            q: asyncio.Queue = asyncio.Queue()

            # Collect assistant response
            assistant_response: List[str] = []

            def on_chunk(chunk: dict) -> None:
                """Callback to handle response chunks from the streaming service."""
                # Collect assistant response text
                if chunk.get("type") == "delta":
                    assistant_response.append(chunk.get("text", ""))

                def _put() -> None:
                    try:
                        # 將背景執行緒中的 chunk 投遞回 asyncio 事件迴圈
                        q.put_nowait(chunk)
                    except asyncio.QueueFull:
                        logger.warning(f"Queue full for session {session_id}")

                loop.call_soon_threadsafe(_put)

            # Use asyncio task instead of threading for better resource management
            task = loop.run_in_executor(
                None,
                request_stream_sync,
                user_msg,
                model,
                on_chunk,
                history.copy()
            )

            try:
                while True:
                    # 從 queue 逐段取出模型回覆並轉發給前端
                    chunk = await q.get()
                    await websocket.send_text(json_dumps(chunk))

                    if chunk.get("type") in ("done", "error"):
                        # Update history after conversation completes
                        if chunk.get("type") == "done" and assistant_response:
                            # 僅在正常完成時，把 user/assistant 內容寫入歷史
                            history.append({"role": "user", "content": user_msg})
                            history.append({"role": "assistant", "content": "".join(assistant_response)})

                            # Limit history length (keep most recent messages)
                            if len(history) > settings.history_max_length:
                                conversation_sessions[session_id] = history[-settings.history_max_length:]
                            
                            logger.info(f"Conversation completed for session {session_id}, history size: {len(history)}")
                        break

            except Exception as e:
                logger.error(f"Error processing response for session {session_id}: {e}")
                await websocket.send_text(json_dumps({
                    "type": "error",
                    "error": "處理回應時發生錯誤"
                }))
            finally:
                # Ensure task is completed
                # 等待背景工作結束，避免 executor 任務殘留
                await task

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session_id={session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler for session {session_id}: {e}")
    finally:
        # Cleanup conversation history for this connection
        if session_id in conversation_sessions:
            del conversation_sessions[session_id]
            logger.info(f"Session cleaned up: session_id={session_id}")
