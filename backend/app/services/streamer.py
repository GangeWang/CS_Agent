import json
import logging
import re
from typing import AsyncIterator, Callable, Optional

import httpx
from ..config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 80

# 基礎客服人設
SYSTEM_PROMPT = (
    "你是服務中心線上客服助手。"
    "請全程使用自然、有人味的繁體中文。"
    "不要提到 ChatGPT、OpenAI、AI、語言模型。"
    "若被問「你是誰」，請固定回答："
    "「您好，我是服務中心的線上客服助手，很高興為您服務。」"
    "\n"
    "重要：請勿輸出推理過程、analysis、工具呼叫內容或任何內部標記；"
    "只輸出給使用者看的最終回覆內容。"
)

# ── Channel filter 常數 ───────────────────────────────────────────────
_CHANNEL_FINAL_MARKER = "<|channel|>final<|message|>"
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_ANALYSIS_PREFIX_RE = re.compile(r"^\s*(analysis|assistantcommentary)\b", re.IGNORECASE)


# ── 動態 endpoint ────────────────────────────────────────────────────
def _get_endpoints():
    base = settings.llama_api_url.rstrip("/")
    return f"{base}/api/stream", f"{base}/api/generate"


# ── Channel 過濾器 ──────────────────────────────────────────────────
class _ChannelFilter:
    def __init__(self, on_chunk: Callable[[dict], None]) -> None:
        self._on_chunk = on_chunk
        self._found_final = False
        self._buf = ""

    def feed(self, chunk: dict) -> None:
        ctype = chunk.get("type")
        if ctype == "done":
            if not self._found_final and self._buf:
                # 從未出現 final marker：buffer 可能是 analysis/CoT 內容（marker 被 chunk 邊界切斷）。
                # 只要 buffer 內含任何特殊標記語法就整段丟棄，絕不 flush 給使用者；
                # 純文字（無標記語法，如非 chat-template 模型輸出）才允許 flush。
                if _SPECIAL_TOKEN_RE.search(self._buf):
                    logger.warning("Dropped %d chars of buffered text containing special tokens (no final marker)", len(self._buf))
                else:
                    clean = _sanitize_visible_text(self._buf.strip())
                    if clean:
                        self._on_chunk({"type": "delta", "text": clean})
            self._buf = ""
            self._on_chunk(chunk)
            return
        if ctype != "delta":
            self._on_chunk(chunk)
            return

        text = chunk.get("text", "")
        if not text:
            return

        if self._found_final:
            clean = _SPECIAL_TOKEN_RE.sub("", text)
            clean = _sanitize_visible_text(clean)
            if clean:
                self._on_chunk({"type": "delta", "text": clean})
            return

        self._buf += text
        if _CHANNEL_FINAL_MARKER in self._buf:
            _, after = self._buf.split(_CHANNEL_FINAL_MARKER, 1)
            self._found_final = True
            self._buf = ""
            clean = _SPECIAL_TOKEN_RE.sub("", after)
            clean = _sanitize_visible_text(clean)
            if clean:
                self._on_chunk({"type": "delta", "text": clean})


def _sanitize_visible_text(text: str) -> str:
    if not text:
        return ""
    if _ANALYSIS_PREFIX_RE.match(text):
        return ""
    return text


def _strip_channel_tokens(text: str) -> str:
    if _CHANNEL_FINAL_MARKER in text:
        _, after = text.split(_CHANNEL_FINAL_MARKER, 1)
        text = after
    text = _SPECIAL_TOKEN_RE.sub("", text).strip()
    return _sanitize_visible_text(text).strip()


def _build_effective_system_prompt(conversation_history: Optional[list[dict]]) -> str:
    if not conversation_history:
        return SYSTEM_PROMPT

    system_parts = [
        msg.get("content", "").strip()
        for msg in conversation_history
        if msg.get("role") == "system" and msg.get("content", "").strip()
    ]
    if system_parts:
        return f"{SYSTEM_PROMPT}\n\n" + "\n\n".join(system_parts)
    return SYSTEM_PROMPT


def _debug(*args) -> None:
    if getattr(settings, "llama_debug", False):
        logger.debug(" ".join(str(arg) for arg in args))


def _extract_text_from_part(part: dict) -> Optional[str]:
    if isinstance(part.get("text"), str):
        return part.get("text")
    choices = part.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                cont = delta.get("content") or delta.get("text")
                if isinstance(cont, str) and cont:
                    return cont
            if isinstance(first.get("text"), str):
                return first.get("text")
    message = part.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message.get("content")
    return None


# ── HTTP client 重用 ─────────────────────────────────────────────────
_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=settings.llama_request_timeout,
                connect=settings.connect_timeout,
            )
        )
    return _client


async def aclose_http_client() -> None:
    """應用程式關閉時呼叫（main.py lifespan），釋放連線池。"""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _build_request(user_msg: str, conversation_history: Optional[list[dict]]) -> tuple[dict, dict]:
    effective_system_prompt = _build_effective_system_prompt(conversation_history)
    messages = [m for m in (conversation_history or []) if m.get("role") != "system"]
    messages.append({"role": "user", "content": user_msg})

    max_tokens = getattr(settings, "llama_max_tokens", 512)
    # 保留一個小下限避免 0/負值，但尊重設定值（先前硬性 max(…, 1024) 讓 LLAMA_MAX_TOKENS=512 永遠無效）
    effective_max_tokens = max(256, int(max_tokens))

    payload = {
        "prompt": user_msg,
        "max_tokens": effective_max_tokens,
        "system_prompt": effective_system_prompt,
        "messages": messages,
    }
    headers = {"Accept": "text/event-stream, application/json"}
    if settings.llama_api_key:
        headers["X-API-KEY"] = settings.llama_api_key
    return payload, headers


def _parse_stream_line(raw: str) -> list[dict]:
    """把推理服務的一行 SSE 轉成 0～2 個 chunk（delta / done / error）。"""
    line = raw.strip()
    _debug("RAW LINE:", line)
    data = line[len("data:"):].strip() if line.startswith("data:") else line
    if not data:
        return []
    if data == "[DONE]":
        return [{"type": "done"}]

    try:
        part = json.loads(data)
    except Exception:
        _debug("Dropped non-JSON SSE line:", data[:200])
        return []

    # 上游回傳 {"error": …} 時要明確終止並回通用錯誤（否則只能等 chunk 逾時）
    if isinstance(part, dict) and part.get("error"):
        logger.warning("LLM stream error payload: %s", json.dumps(part, ensure_ascii=False)[:500])
        return [{"type": "error", "error": "LLM 服務發生錯誤，請稍後再試。"}]

    text_chunk = _extract_text_from_part(part) if isinstance(part, dict) else None
    text_chunk = _sanitize_visible_text(text_chunk or "")
    out = [{"type": "delta", "text": text_chunk}] if text_chunk else []
    if isinstance(part, dict) and part.get("done") is True:
        out.append({"type": "done"})
    elif not out:
        _debug("skipping non-visible chunk:", json.dumps(part, ensure_ascii=False)[:200])
    return out


# ── Streaming ───────────────────────────────────────────────────────
async def stream_chat(
    user_msg: str,
    model: Optional[str],
    conversation_history: Optional[list[dict]] = None,
) -> AsyncIterator[dict]:
    """
    以 async httpx 串流呼叫推理服務，逐段 yield delta / done / error（已經過 channel 過濾）。

    取消：呼叫端停止迭代（aclose）或所在的 task 被取消時，`async with client.stream(...)` 會關閉
    上游連線，推理服務偵測到斷線就停止生成，不再佔用推理名額。不需要執行緒或 cancel_event。

    model 目前不會送到推理服務（服務端只載入一個模型），保留參數供日後多模型使用。
    """
    pending: list[dict] = []
    channel_filter = _ChannelFilter(pending.append)
    # 是否已有任何 chunk 送到下游：一旦送出過內容，就不能再走非串流 fallback
    # （否則使用者會看到「前半段 + 完整新回覆」重複輸出）
    state = {"emitted": False, "finished": False}

    def emit(chunk: dict) -> None:
        state["emitted"] = True
        if chunk.get("type") in ("done", "error"):
            state["finished"] = True
        channel_filter.feed(chunk)

    payload, headers = _build_request(user_msg, conversation_history)
    client = _get_http_client()
    stream_url, once_url = _get_endpoints()

    try:
        async with client.stream("POST", stream_url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = (await resp.aread())[:500]
                # 細節（含上游 response body）只進 log；對外僅回通用訊息，避免洩漏內部拓撲
                logger.warning("LLM stream endpoint HTTP %s: %r", resp.status_code, body)
                emit({"type": "error", "error": f"LLM 服務暫時無法回應（HTTP {resp.status_code}）"})
            else:
                async for raw in resp.aiter_lines():
                    for chunk in _parse_stream_line(raw):
                        emit(chunk)
                        if state["finished"]:
                            break
                    while pending:
                        yield pending.pop(0)
                    if state["finished"]:
                        break
                if not state["finished"]:
                    emit({"type": "done"})
        while pending:
            yield pending.pop(0)
        return
    except Exception as e:
        if state["emitted"]:
            # 串流中途斷線：已送出部分內容，不可重新生成（會重複輸出），直接回報錯誤
            logger.warning("Stream broke after partial output, no fallback: %s", e)
            while pending:
                yield pending.pop(0)
            yield {"type": "error", "error": "LLM 串流中斷，請稍後再試。"}
            return
        logger.warning("Stream exception before any output, fallback to non-streaming: %s", e)

    # ── 非串流 fallback（串流在送出任何內容前就失敗） ──
    # 完整生成可遠超 llama_request_timeout，使用 llama_stream_timeout
    try:
        resp2 = await client.post(once_url, json=payload, headers=headers, timeout=settings.llama_stream_timeout)
    except Exception as e2:
        logger.warning("LLAMA non-streaming fallback failed: %s", e2)
        yield {"type": "error", "error": "LLM 連線失敗，請稍後再試。"}
        return

    if resp2.status_code != 200:
        logger.warning("LLM generate endpoint HTTP %s: %s", resp2.status_code, resp2.text[:500])
        yield {"type": "error", "error": f"LLM 服務暫時無法回應（HTTP {resp2.status_code}）"}
        return

    try:
        j = resp2.json()
        text = j.get("text") or (j.get("message") and j["message"].get("content")) or None
        if not text:
            text = json.dumps(j, ensure_ascii=False)
    except Exception:
        text = resp2.text or ""

    text = _strip_channel_tokens(text) if text else ""
    if not text:
        yield {"type": "error", "error": "模型回傳但無文字"}
        return

    for i in range(0, len(text), CHUNK_SIZE):
        yield {"type": "delta", "text": text[i : i + CHUNK_SIZE]}
    yield {"type": "done"}


async def generate_text(
    prompt: str,
    model: Optional[str] = None,
    conversation_history: Optional[list[dict]] = None,
) -> tuple[str, Optional[str]]:
    """跑完一次串流，回傳 (完整文字, 錯誤訊息或 None)。供摘要、上下文壓縮等不需逐段轉送的地方使用。"""
    parts: list[str] = []
    error: Optional[str] = None
    async for chunk in stream_chat(prompt, model, conversation_history):
        if chunk.get("type") == "delta":
            parts.append(chunk.get("text", ""))
        elif chunk.get("type") == "error":
            error = chunk.get("error", "")
    return "".join(parts).strip(), error
