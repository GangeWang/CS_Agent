import json
import logging
import re
from typing import Callable, Optional

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
                clean = _SPECIAL_TOKEN_RE.sub("", self._buf).strip()
                clean = _sanitize_visible_text(clean)
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
    if getattr(settings, "ollama_debug", False):
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
_client: Optional[httpx.Client] = None


def _get_http_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=httpx.Timeout(
                timeout=settings.llama_request_timeout,
                connect=settings.connect_timeout,
            )
        )
    return _client


# ── Streaming wrapper ───────────────────────────────────────────────
def request_stream_sync(
    user_msg: str,
    model: Optional[str],
    on_chunk: Callable[[dict], None],
    conversation_history: Optional[list[dict]] = None,
) -> None:
    _filter = _ChannelFilter(on_chunk)
    filtered_on_chunk = _filter.feed

    effective_system_prompt = _build_effective_system_prompt(conversation_history)
    messages = [m for m in (conversation_history or []) if m.get("role") != "system"]
    messages.append({"role": "user", "content": user_msg})

    max_tokens = getattr(settings, "llama_max_tokens", 512)
    effective_max_tokens = max(max_tokens, 1024)

    payload = {
        "prompt": user_msg,
        "max_tokens": effective_max_tokens,
        "system_prompt": effective_system_prompt,
        "messages": messages,
    }

    headers = {"Accept": "text/event-stream, application/json"}
    if settings.llama_api_key:
        headers["X-API-KEY"] = settings.llama_api_key

    client = _get_http_client()
    stream_url, once_url = _get_endpoints()

    # ── streaming 主流程 ──
    try:
        with client.stream("POST", stream_url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                filtered_on_chunk({"type": "error", "error": f"HTTP {resp.status_code}: {resp.text}"})
                return

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
                line = line.strip()
                _debug("RAW LINE:", line)

                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                else:
                    data = line

                if not data:
                    continue
                if data == "[DONE]":
                    filtered_on_chunk({"type": "done"})
                    return

                try:
                    part = json.loads(data)
                except Exception:
                    _debug("Dropped non-JSON SSE line:", data[:200])
                    continue

                if isinstance(part, dict) and part.get("done") is True:
                    text_chunk = _extract_text_from_part(part) or ""
                    text_chunk = _sanitize_visible_text(text_chunk)
                    if text_chunk:
                        filtered_on_chunk({"type": "delta", "text": text_chunk})
                    filtered_on_chunk({"type": "done"})
                    return

                text_chunk = _extract_text_from_part(part)
                if not text_chunk:
                    _debug("skipping non-visible chunk:", json.dumps(part, ensure_ascii=False)[:200])
                    continue

                text_chunk = _sanitize_visible_text(text_chunk)
                if not text_chunk:
                    _debug("dropped suspicious/empty chunk")
                    continue

                filtered_on_chunk({"type": "delta", "text": text_chunk})

            filtered_on_chunk({"type": "done"})
            return

    except Exception as e:
        logger.warning("Stream exception, fallback to non-streaming: %s", e)

        try:
            resp2 = client.post(once_url, json=payload, headers=headers, timeout=settings.llama_request_timeout)
        except Exception as e2:
            filtered_on_chunk({"type": "error", "error": f"LLAMA 連線失敗：{e2}"})
            return

        if resp2.status_code != 200:
            filtered_on_chunk({"type": "error", "error": f"HTTP {resp2.status_code}: {resp2.text}"})
            return

        try:
            j = resp2.json()
            text = j.get("text") or (j.get("message") and j["message"].get("content")) or None
            if not text:
                text = json.dumps(j, ensure_ascii=False)
        except Exception:
            text = resp2.text or ""

        if not text:
            filtered_on_chunk({"type": "error", "error": "模型回傳但無文字"})
            return

        text = _strip_channel_tokens(text)
        if not text:
            filtered_on_chunk({"type": "error", "error": "模型回傳但無文字（過濾後為空）"})
            return

        for i in range(0, len(text), CHUNK_SIZE):
            on_chunk({"type": "delta", "text": text[i : i + CHUNK_SIZE]})
        on_chunk({"type": "done"})
        return