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
# gpt-oss（harmony 格式）的輸出：先 analysis channel（思考），再 final channel（給使用者看的回覆）。
# llama-cpp-python 轉文字時預設**不印特殊標記**（detokenize(special=False)），所以同一段輸出可能是：
#   有標記：<|channel|>analysis<|message|>思考…<|end|><|start|>assistant<|channel|>final<|message|>回覆
#   無標記：analysis思考…assistantfinal回覆
# 推理服務會先拿掉 <|start|>/<|end|>/<|return|>。兩種寫法都要能切出 final 之後的內容。
_CHANNEL_FINAL_MARKER = "<|channel|>final<|message|>"
_FINAL_MARKERS = (_CHANNEL_FINAL_MARKER, "assistantfinal")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
# 輸出開頭是這些字 → harmony 格式（先緩衝，等 final 才送出）；其他開頭 → 一般純文字，直接串流
_HARMONY_STARTS = ("<|", "analysis", "commentary", "assistant", "final")
NO_FINAL_ERROR = "模型這次沒有產生回覆，請再試一次。"


# ── 動態 endpoint ────────────────────────────────────────────────────
def _get_endpoints():
    base = settings.llama_api_url.rstrip("/")
    return f"{base}/api/stream", f"{base}/api/generate"


# ── Channel 過濾器 ──────────────────────────────────────────────────
class _ChannelFilter:
    """
    把推理服務的串流切成「給使用者看的部分」。

    mode：
    - None：還在看開頭，判斷不出格式（例如只收到 "ana"）
    - "plain"：開頭不是 harmony 的字樣 → 一般模型輸出，收到就送
    - "harmony"：還在 analysis/commentary channel → 只緩衝，不送
    - "final"：已經看到 final 標記 → 之後的內容逐段送出

    harmony 格式到結束都沒看到 final（例如思考把 max_tokens 用完），緩衝內容一律丟掉、改回錯誤，
    絕不把思考內容送給使用者。
    """

    def __init__(self, on_chunk: Callable[[dict], None]) -> None:
        self._on_chunk = on_chunk
        self._mode: Optional[str] = None
        self._buf = ""

    def _emit_text(self, text: str) -> None:
        clean = _SPECIAL_TOKEN_RE.sub("", text)
        if clean:
            self._on_chunk({"type": "delta", "text": clean})

    def _try_enter_final(self) -> bool:
        hits = [(self._buf.find(m), m) for m in _FINAL_MARKERS if m in self._buf]
        head = self._buf.lstrip()
        if not hits and head.startswith("final") and not head.startswith("finally"):
            # 無標記且沒有 analysis，直接以 final channel 開頭："final您好…"
            hits = [(len(self._buf) - len(head), "final")]
        if not hits:
            return False
        pos, marker = min(hits)
        after = self._buf[pos + len(marker):]
        self._mode = "final"
        self._buf = ""
        self._emit_text(after.lstrip() if marker != _CHANNEL_FINAL_MARKER else after)
        return True

    def feed(self, chunk: dict) -> None:
        ctype = chunk.get("type")
        if ctype == "done":
            self._finish(chunk)
            return
        if ctype != "delta":
            self._on_chunk(chunk)
            return

        text = chunk.get("text", "")
        if not text:
            return
        if self._mode in ("final", "plain"):
            self._emit_text(text)
            return

        self._buf += text
        if self._mode is None:
            head = self._buf.lstrip()
            if not head:
                return
            if head.startswith(_HARMONY_STARTS):
                self._mode = "harmony"
            elif any(s.startswith(head) for s in _HARMONY_STARTS):
                return  # 只收到 "ana" 這種前綴，等下一段再判斷
            else:
                self._mode = "plain"
                buf, self._buf = self._buf, ""
                self._emit_text(buf)
                return
        self._try_enter_final()

    def _finish(self, done_chunk: dict) -> None:
        if self._buf and self._mode is None:
            # 整段只有一個很短的前綴（例如回覆就是 "fin"）：當純文字送出
            self._emit_text(self._buf)
        elif self._mode == "harmony":
            if self._buf:
                logger.warning(
                    "Dropped %d chars of model output without a final channel (reasoning only?)", len(self._buf)
                )
            self._buf = ""
            self._on_chunk({"type": "error", "error": NO_FINAL_ERROR})
            return
        self._buf = ""
        self._on_chunk(done_chunk)


def _strip_channel_tokens(text: str) -> str:
    """非串流的完整輸出：用同一個過濾器切出 final 內容；切不出來（只有思考）回空字串。"""
    out: list[dict] = []
    f = _ChannelFilter(out.append)
    f.feed({"type": "delta", "text": text})
    f.feed({"type": "done"})
    if any(c.get("type") == "error" for c in out):
        return ""
    return "".join(c.get("text", "") for c in out if c.get("type") == "delta").strip()


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

    # 不在這裡逐段丟掉 "analysis" 開頭的片段：_ChannelFilter 要靠它判斷是不是 harmony 格式
    text_chunk = _extract_text_from_part(part) if isinstance(part, dict) else None
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
