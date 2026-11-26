"""
WebSocket /ws/chat：一條連線 = 一個 ChatSession。

每則使用者訊息的流程（ChatSession._on_user_message）：
  速率限制 → guardrail 分類 → RAG 查詢 → 組 system 指令 → 上下文管理 → 串流回覆 → 寫入歷史
其他訊息類型：ping（含閒置警告/逾時結束）、clear_history、end_conversation。

收訊迴圈和「回覆一則訊息」分開跑：回覆在背景 task（同一時間最多一則），收訊迴圈在產生回覆時
仍然會處理 ping，所以前端的心跳不會因為回覆太久而把連線切斷。end_conversation / clear_history
在回覆途中送來時，會先取消這次生成。

前端安全規則：
- 只把 delta / done / error 這三種串流 chunk 送給瀏覽器（另有 guardrail、pong 等控制事件）。
- 除錯資訊一律只進後端 log，不送到 WS。
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from collections import deque
from typing import AsyncIterator, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..services import inference_info, session_store
from ..services.agent.runner import run_agent_stream
from ..services.context_manager import manage_context, summarize_conversation
from ..services.guardrail import classify_text
from ..services.instructions import REFUSED_LABELS, build_guardrail_instruction, build_rag_instruction
from ..services.rag import retrieve_rag_hit
from ..services.streamer import stream_chat
from ..utils.jsonsafe import json_dumps
from ..utils.messages import extract_latest_user_message

router = APIRouter()
logger = logging.getLogger(__name__)

IDLE_TIMEOUT_SECONDS = 180
IDLE_WARNING_SECONDS_BEFORE_END = 60

# Only these chunk types are allowed to reach the browser UI.
ALLOWED_STREAM_TYPES = {"delta", "done", "error"}


def _origin_allowed(websocket: WebSocket) -> bool:
    """
    WebSocket handshake Origin 檢查（CSWSH 防護）。

    CORS middleware 不涵蓋 WebSocket，需在 accept 前自行驗證：
    1) 無 Origin header（非瀏覽器客戶端：curl / 測試 / 伺服器間呼叫）→ 允許
       （CSWSH 攻擊只能來自瀏覽器，瀏覽器一定帶 Origin）
    2) 白名單含 "*" → 允許（明確關閉檢查）
    3) Origin 的 host:port 與 Host header 相同（同源反代：nginx / vite proxy）→ 允許
    4) Origin 在 settings.ws_allowed_origins 白名單 → 允許
    其餘拒絕。
    """
    origin = websocket.headers.get("origin")
    if not origin:
        return True

    allowed = [o for o in (settings.ws_allowed_origins or []) if o]
    if "*" in allowed:
        return True

    host_header = (websocket.headers.get("host") or "").lower()
    try:
        origin_netloc = urlparse(origin).netloc.lower()
    except Exception:
        origin_netloc = ""

    if host_header and origin_netloc and origin_netloc == host_header:
        return True

    if origin in allowed or origin_netloc in {a.lower() for a in allowed}:
        return True

    return False


async def _pump_stream(
    websocket,
    session_id: str,
    chunks: AsyncIterator[dict],
) -> Tuple[List[str], str]:
    """
    把 async 串流的 delta / error 轉送到 WS，回傳 (assistant 文字片段, status)。

    websocket 只需要有 `async send_text(str)`。
    **done 不在這裡送**：呼叫端先把這輪寫進歷史，再送 done。這樣前端一收到 done 就送
    end_conversation 時，摘要一定含這一輪。

    status：
    - "done" / "error"：上游正常結束或回報錯誤
    - "timeout"：兩個 chunk 間隔超過 stream_chunk_timeout。wait_for 會取消等待中的 __anext__，
      串流的 `async with` 隨之關閉上游連線，推理服務停止生成
    使用者斷線（送出失敗）或協程被取消時，finally 的 aclose() 同樣會關閉上游連線，再把例外往外拋。
    """
    texts: List[str] = []
    it = chunks.__aiter__()
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(it.__anext__(), timeout=settings.stream_chunk_timeout)
            except StopAsyncIteration:
                # 串流結束卻沒送 done/error
                await websocket.send_text(json_dumps({"type": "error", "error": "LLM 任務執行失敗"}))
                return texts, "error"
            except asyncio.TimeoutError:
                logger.error("Streaming timeout for session %s; generation cancelled", session_id)
                await websocket.send_text(json_dumps({"type": "error", "error": "模型回應逾時"}))
                return texts, "timeout"
            except Exception as e:  # 上游未捕捉的例外：細節只進 log
                logger.error("Stream failed for session %s: %r", session_id, e)
                await websocket.send_text(json_dumps({"type": "error", "error": "LLM 任務執行失敗"}))
                return texts, "error"

            t = chunk.get("type")
            if t not in ALLOWED_STREAM_TYPES:
                continue  # Drop any internal/debug chunks
            if t == "done":
                return texts, "done"
            if t == "delta":
                texts.append(chunk.get("text", ""))
            await websocket.send_text(json_dumps(chunk))
            if t == "error":
                return texts, "error"
    finally:
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # 已被取消或已結束的 generator
                pass


class ChatSession:
    """一條 WebSocket 連線的對話狀態，以及各種訊息類型的處理。"""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        # uuid 取代 id(websocket)：日誌可追蹤，且不受物件位址重用影響
        self.session_id = uuid.uuid4().hex
        self._loop = asyncio.get_running_loop()
        self._last_dialogue_at = self._loop.time()
        self._idle_warning_sent = False
        self._last_model: Optional[str] = None
        self._msg_times: deque = deque()  # 速率限制：window 內的使用者訊息時間戳
        # 正在回覆的那一則訊息（背景 task）；同一時間最多一則
        self._turn: Optional[asyncio.Task] = None
        # 這一輪的串流是否還在進行；串流結束後 task 只剩收尾（寫歷史、送 done），下一則可以等它
        self._streaming = False
        # 收訊迴圈（pong 等）和回覆 task 會同時送訊息，用鎖確保一次只寫一個 frame
        self._send_lock = asyncio.Lock()

    async def send_text(self, text: str) -> None:
        async with self._send_lock:
            await self.ws.send_text(text)

    async def send(self, payload: dict) -> None:
        await self.send_text(json_dumps(payload))

    def _busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def _cancel_turn(self) -> None:
        """取消進行中的回覆（關閉上游連線，推理服務停止生成），並等它收尾。"""
        turn = self._turn
        self._turn = None
        if turn is not None and not turn.done():
            turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)

    async def run(self) -> None:
        session_store.create(self.session_id)
        logger.info("WebSocket connection established: session_id=%s", self.session_id)
        try:
            while True:
                payload = await self._receive()
                if payload is not None and not await self._dispatch(payload):
                    break
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: session_id=%s", self.session_id)
        except Exception as e:
            logger.exception("Unexpected error in WebSocket handler for session %s: %s", self.session_id, e)
        finally:
            await self._cancel_turn()
            session_store.drop(self.session_id)
            logger.info("Session cleaned up: session_id=%s", self.session_id)

    # ---------- 收訊與分派 ----------
    async def _receive(self) -> Optional[dict]:
        """收一則訊息並解析成 dict；太大或格式錯誤時回錯誤給前端並回傳 None。"""
        raw = await self.ws.receive_text()

        # max_message_size 以 bytes 計（UTF-8）；中文字元 3 bytes，用字元數會低估 3 倍
        if len(raw.encode("utf-8")) > settings.max_message_size:
            await self.send({"type": "error", "error": f"訊息過大 (最大 {settings.max_message_size} bytes)"})
            return None

        try:
            payload = json.loads(raw)
        except Exception as e:
            logger.warning("Invalid JSON from session %s: %s", self.session_id, e)
            payload = None
        if not isinstance(payload, dict):
            await self.send({"type": "error", "error": "JSON 格式不正確"})
            return None
        return payload

    async def _dispatch(self, payload: dict) -> bool:
        """處理一則訊息；回傳 False 代表對話已結束、連線要關閉。"""
        kind = payload.get("type")
        if kind == "ping":
            return await self._on_ping()
        if kind == "clear_history":
            await self._cancel_turn()
            await self._on_clear_history()
            return True
        if kind == "end_conversation":
            await self._cancel_turn()
            await self.end_conversation("manual")
            return False
        if self._busy():
            if self._streaming:
                await self.send({"type": "error", "error": "上一則訊息還在回覆中，請稍候。"})
                return True
            # 上一輪已送出 done / error，只差收尾：等它結束再處理這一則
            await asyncio.gather(self._turn, return_exceptions=True)
        self._streaming = True
        self._turn = asyncio.create_task(self._run_turn(payload))
        return True

    async def _run_turn(self, payload: dict) -> None:
        try:
            await self._on_user_message(payload)
        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect:
            pass  # 收訊迴圈會收到斷線並清理
        except Exception as e:
            logger.exception("Turn failed for session %s: %s", self.session_id, e)
        finally:
            self._streaming = False
            # 回覆結束才開始算閒置：回覆花很久時，不會一結束就被當成閒置太久
            self._mark_active()

    # ---------- 控制訊息 ----------
    def _mark_active(self) -> None:
        self._last_dialogue_at = self._loop.time()
        self._idle_warning_sent = False

    async def _on_ping(self) -> bool:
        if self._busy():
            # 正在回覆不算閒置
            await self.send({"type": "pong"})
            return True
        idle_elapsed = self._loop.time() - self._last_dialogue_at
        if idle_elapsed > IDLE_TIMEOUT_SECONDS:
            await self.end_conversation("idle_timeout")
            return False

        warn_from = IDLE_TIMEOUT_SECONDS - IDLE_WARNING_SECONDS_BEFORE_END
        if not self._idle_warning_sent and warn_from <= idle_elapsed < IDLE_TIMEOUT_SECONDS:
            remaining_seconds = max(0, math.ceil(IDLE_TIMEOUT_SECONDS - idle_elapsed))
            await self.send({"type": "idle_warning", "remaining_seconds": remaining_seconds})
            self._idle_warning_sent = True

        await self.send({"type": "pong"})
        return True

    async def _on_clear_history(self) -> None:
        session_store.clear(self.session_id)
        self._mark_active()
        await self.send({"type": "history_cleared"})

    async def end_conversation(self, reason: str) -> None:
        summary = await summarize_conversation(session_store.get_history(self.session_id), self._last_model)
        # 摘要含對話內容（可能含 PII）：log 只留前 300 字供排查，完整內容不進日誌
        logger.info(
            "Conversation summary session_id=%s reason=%s summary=%s", self.session_id, reason, summary[:300]
        )
        await self.send({"type": "conversation_summary", "reason": reason, "summary": summary})
        await self.send({"type": "conversation_ended", "reason": reason})
        await self.ws.close(code=1000, reason="conversation ended")

    # ---------- 使用者訊息 ----------
    def _rate_limited(self) -> bool:
        # 每則使用者訊息都會觸發 guardrail 推論 + RAG + LLM 生成，需限制頻率
        now = self._loop.time()
        while self._msg_times and now - self._msg_times[0] > settings.ws_rate_limit_window_seconds:
            self._msg_times.popleft()
        if len(self._msg_times) >= settings.ws_rate_limit_messages:
            return True
        self._msg_times.append(now)
        return False

    async def _classify(self, user_msg: str) -> Tuple[str, bool]:
        """回傳 (label, degraded)，並把結果送給前端更新狀態列。"""
        # classify_text 內含 transformer 推論（同步），放到執行緒池避免阻塞 event loop
        result = await asyncio.to_thread(classify_text, user_msg)
        label = result.get("label", "NORMAL")
        degraded = not result.get("available", True)
        if degraded:
            logger.warning(
                "Guardrail DEGRADED for session %s: stage=%s reason=%s",
                self.session_id, result.get("stage"), result.get("reason"),
            )
        # 前端只用來更新狀態列，不會當成對話泡泡顯示
        await self.send({"type": "guardrail", "label": label, "degraded": degraded})
        return label, degraded

    async def _rag_instruction(self, user_msg: str) -> str:
        # 同步檔案 I/O + 分詞計分，包 to_thread 避免阻塞 event loop
        rag_hit = await asyncio.to_thread(retrieve_rag_hit, user_msg)
        if rag_hit:
            logger.info(
                "RAG HIT session_id=%s score=%s matched_question=%s",
                self.session_id, rag_hit.score, rag_hit.matched_question,
            )
        else:
            logger.info("RAG MISS session_id=%s", self.session_id)
        return build_rag_instruction(rag_hit)

    async def _managed_history(
        self, system_instruction: str, user_msg: str, model: Optional[str], strategy: Optional[str]
    ) -> List[dict]:
        # 還不知道推理服務的 n_ctx 時（例如推理服務比後端晚啟動）補問一次，讓上下文預算對齊實際模型
        await inference_info.refresh_if_stale()
        # compress 策略會呼叫 LLM 做摘要（async，不阻塞 event loop）
        history, meta = await manage_context(
            session_store.get_history(self.session_id),
            [{"role": "system", "content": system_instruction}],
            user_msg,
            model,
            strategy,
        )
        session_store.set_history(self.session_id, history)
        if meta.get("applied"):
            logger.info("context managed session_id=%s meta=%s", self.session_id, meta)
        return history

    async def _on_user_message(self, payload: dict) -> None:
        user_msg = extract_latest_user_message(payload.get("messages", []))
        if not user_msg:
            await self.send({"type": "error", "error": "缺少使用者訊息"})
            return

        self._mark_active()
        if self._rate_limited():
            logger.warning("Rate limit exceeded for session %s", self.session_id)
            await self.send({"type": "error", "error": "傳送過於頻繁，請稍候再試。"})
            return

        label, degraded = await self._classify(user_msg)
        parts = [build_guardrail_instruction(label, degraded)]
        if label not in REFUSED_LABELS:
            parts.append(await self._rag_instruction(user_msg))
        system_msg = {"role": "system", "content": "\n\n".join(parts)}

        model = payload.get("model")
        model = model.strip() if isinstance(model, str) and model.strip() else None
        if model:
            self._last_model = model
        strategy = payload.get("context_strategy")
        strategy = strategy if isinstance(strategy, str) else None

        history = await self._managed_history(system_msg["content"], user_msg, model, strategy)

        # 攻擊 / 廣告一律走一般對話（帶拒答指令），不跑 agent 的工具
        if payload.get("mode") == "agent" and label not in REFUSED_LABELS:
            chunks = run_agent_stream([system_msg, *history, {"role": "user", "content": user_msg}], model)
        else:
            chunks = stream_chat(user_msg, model, [system_msg, *history])

        try:
            texts, status = await _pump_stream(self, self.session_id, chunks)
        finally:
            self._streaming = False
        if status != "done":
            return
        reply = "".join(texts).strip()
        # 被拒答的訊息（攻擊 / 廣告）不寫進歷史：guardrail 只看最新一則，
        # 如果歷史留著攻擊內容，下一句「照上面說的做」就會被判成 NORMAL、模型卻讀得到攻擊。
        if reply and label not in REFUSED_LABELS:
            session_store.append_turn(self.session_id, user_msg, reply)
        await self.send({"type": "done"})


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    # Origin 檢查需在 accept 之前：close() 會以 HTTP 403 拒絕 handshake
    if not _origin_allowed(websocket):
        logger.warning(
            "WebSocket rejected (origin not allowed): origin=%s host=%s",
            websocket.headers.get("origin"),
            websocket.headers.get("host"),
        )
        await websocket.close(code=1008, reason="origin not allowed")
        return

    await websocket.accept()
    await ChatSession(websocket).run()
