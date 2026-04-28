import json
import logging
import re
from typing import Callable, Optional
import httpx

from ..config import settings

logger = logging.getLogger(__name__)

ENDPOINT_STREAM = settings.llama_api_url.rstrip("/") + "/api/stream"
ENDPOINT_ONCE = settings.llama_api_url.rstrip("/") + "/api/generate"

CHUNK_SIZE = 80

# 基礎客服人設：固定不變的身份設定
SYSTEM_PROMPT = (
    "你是服務中心線上客服助手。"
    "請全程使用自然、有人味的繁體中文。"
    "不要提到 ChatGPT、OpenAI、AI、語言模型。"
    "若被問「你是誰」，請固定回答："
    "「您好，我是服務中心的線上客服助手，很高興為您服務。」"
)

# ── Channel filter 常數 ────────────────────────────────────────────────────────
# 部分指令微調模型會在輸出中插入 channel 標記：
#   <|channel|>analysis<|message|>  → 模型內部推理，不應給使用者看到
#   <|channel|>final<|message|>     → 真正對外回覆，從此處開始放行
# 任何 <|...|> 格式的特殊 token 一律清除，避免其他標記漏出。
_CHANNEL_FINAL_MARKER = "<|channel|>final<|message|>"
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


class _ChannelFilter:
    """
    包裝 on_chunk callback，過濾 LLM 輸出的 channel 標記。

    行為：
    - 找到 <|channel|>final<|message|> 之前的文字暫存於緩衝，不轉發給前端。
    - 找到 marker 後，只放行其後的文字，並清除殘留的 <|...|> token。
    - 串流結束（done）時若從未出現 marker，將整段緩衝清除特殊 token 後一次放行。
      這個 fallback 確保不使用 channel 格式的模型仍能正常輸出與記錄歷史。
    - done / error 類型 chunk 直接穿透。
    """

    def __init__(self, on_chunk: Callable[[dict], None]) -> None:
        self._on_chunk = on_chunk
        self._found_final = False
        # 完整累積緩衝：不提前修剪，確保 fallback 可用
        self._buf = ""

    def feed(self, chunk: dict) -> None:
        chunk_type = chunk.get("type")

        if chunk_type == "done":
            # 串流結束：若從未找到 final marker，將緩衝內容清理後整段輸出（fallback）
            if not self._found_final and self._buf:
                clean = _SPECIAL_TOKEN_RE.sub("", self._buf).strip()
                if clean:
                    self._on_chunk({"type": "delta", "text": clean})
            self._buf = ""
            self._on_chunk(chunk)
            return

        if chunk_type != "delta":
            # error 等其他類型直接放行
            self._on_chunk(chunk)
            return

        text = chunk.get("text", "")
        if not text:
            return

        if self._found_final:
            # 已進入 final 區段：清除殘留特殊 token 後立即放行
            clean = _SPECIAL_TOKEN_RE.sub("", text)
            if clean:
                self._on_chunk({"type": "delta", "text": clean})
            return

        # 尚未找到 final marker：完整累積，不提前修剪
        self._buf += text

        if _CHANNEL_FINAL_MARKER in self._buf:
            _, after = self._buf.split(_CHANNEL_FINAL_MARKER, 1)
            self._found_final = True
            self._buf = ""  # 釋放記憶體
            clean = _SPECIAL_TOKEN_RE.sub("", after)
            if clean:
                self._on_chunk({"type": "delta", "text": clean})
        # else: 繼續累積，等待 marker 出現或串流結束觸發 fallback


def _strip_channel_tokens(text: str) -> str:
    """非串流 fallback 用：一次性從完整文字中移除 channel 標記與特殊 token。"""
    if _CHANNEL_FINAL_MARKER in text:
        _, after = text.split(_CHANNEL_FINAL_MARKER, 1)
        text = after
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def _build_effective_system_prompt(conversation_history: Optional[list[dict]]) -> str:
    """
    從 conversation_history 提取 system role 訊息（guardrail instruction），
    與基礎 SYSTEM_PROMPT 合併後回傳。

    ws.py 將 guardrail instruction 注入 augmented_history[0] 作為 system role，
    這裡負責把它轉為 server 實際會使用的 system_prompt 欄位，
    讓 label（ABUSIVE / PROMPT_ATTACK / SPAM / NORMAL）真正影響模型行為。
    """
    if not conversation_history:
        return SYSTEM_PROMPT
    for msg in conversation_history:
        if msg.get("role") == "system":
            guardrail = msg.get("content", "").strip()
            if guardrail:
                return f"{SYSTEM_PROMPT}\n\n{guardrail}"
    return SYSTEM_PROMPT


def _debug(*args) -> None:
    if getattr(settings, "ollama_debug", False):
        logger.debug(" ".join(str(arg) for arg in args))


def _build_prompt(system_prompt: str, conversation_history: Optional[list[dict]], user_msg: str) -> str:
    parts = [system_prompt.strip(), "\n\n對話：\n"]
    if conversation_history:
        for m in conversation_history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not content:
                continue
            if role == "user":
                parts.append(f"使用者：{content}")
            elif role == "assistant":
                parts.append(f"客服助手：{content}")
            else:
                parts.append(f"{role}: {content}")
    parts.append(f"使用者：{user_msg}")
    parts.append("\n客服助手：")
    return "\n".join(parts)


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

    for key in ("response", "response_text", "output", "content"):
        v = part.get(key)
        if isinstance(v, str) and v:
            return v

    return None


def request_stream_sync(
        user_msg: str,
        model: Optional[str],
        on_chunk: Callable[[dict], None],
        conversation_history: Optional[list[dict]] = None
) -> None:
    """
    同步 streaming wrapper。

    修正：
    1. _ChannelFilter 改為完整累積緩衝 + done 時 fallback 輸出，
       解決「沒有 final marker 時記憶消失、文字全部被丟棄」的問題。
    2. _build_effective_system_prompt 從 conversation_history 提取 guardrail instruction，
       注入 payload 的 system_prompt 欄位，讓 label 真正影響模型行為。
    3. messages 陣列中的 system role 訊息一律移除，統一從 system_prompt 欄位傳遞，
       避免 server 重複或錯誤解析。
    4. effective_max_tokens 確保至少 1024，避免 analysis 段落吃掉 token 配額
       導致真正的回覆在中途被截斷。
    """
    # 包裝 on_chunk，過濾 channel 標記
    _filter = _ChannelFilter(on_chunk)
    filtered_on_chunk = _filter.feed

    # 從 conversation_history 提取 guardrail instruction，合併進 system_prompt
    effective_system_prompt = _build_effective_system_prompt(conversation_history)

    # 建立 messages 陣列：排除 system role（已移至 system_prompt 欄位），加入當前用戶訊息
    messages = [
        m for m in (conversation_history or [])
        if m.get("role") != "system"
    ]
    messages.append({"role": "user", "content": user_msg})

    # channel 格式模型的 analysis 段落本身消耗 token，
    # 若配額太小，真正的回覆會在中途被截斷，因此確保至少 1024。
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

    try:
        client = httpx.Client(
            timeout=httpx.Timeout(
                timeout=settings.llama_request_timeout,
                connect=settings.connect_timeout
            )
        )
    except Exception as e:
        logger.error(f"Failed to create HTTP client: {e}")
        filtered_on_chunk({"type": "error", "error": f"建立 HTTP 客戶端失敗：{e}"})
        return

    try:
        with client.stream("POST", ENDPOINT_STREAM, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code}: {resp.text}"
                logger.error(f"Stream request failed: {error_msg}")
                filtered_on_chunk({"type": "error", "error": error_msg})
                return

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
                line = line.strip()
                _debug("RAW LINE:", line)

                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
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
                    text_chunk = data.strip()
                    if text_chunk:
                        filtered_on_chunk({"type": "delta", "text": text_chunk})
                    continue

                if isinstance(part, dict) and part.get("done") is True:
                    text_chunk = _extract_text_from_part(part) or ""
                    if text_chunk:
                        filtered_on_chunk({"type": "delta", "text": text_chunk})
                    filtered_on_chunk({"type": "done"})
                    return

                text_chunk = _extract_text_from_part(part)
                if not text_chunk:
                    _debug("skipping non-visible chunk:", json.dumps(part, ensure_ascii=False)[:200])
                    continue

                filtered_on_chunk({"type": "delta", "text": text_chunk})

            filtered_on_chunk({"type": "done"})
            return

    except Exception as e:
        logger.warning(f"Stream exception, falling back to non-streaming: {e}")

        try:
            resp2 = client.post(
                ENDPOINT_ONCE,
                json=payload,
                headers=headers,
                timeout=settings.llama_request_timeout
            )
        except Exception as e2:
            error_msg = f"LLAMA 連線失敗：{e2}"
            logger.error(error_msg)
            filtered_on_chunk({"type": "error", "error": error_msg})
            return

        if resp2.status_code != 200:
            error_msg = f"HTTP {resp2.status_code}: {resp2.text}"
            logger.error(error_msg)
            filtered_on_chunk({"type": "error", "error": error_msg})
            return

        try:
            j = resp2.json()
            text = (
                j.get("text")
                or (j.get("message") and j["message"].get("content"))
                or j.get("response")
                or j.get("output")
            )
            if not text:
                text = json.dumps(j, ensure_ascii=False)
        except Exception:
            text = resp2.text or ""

        if not text:
            filtered_on_chunk({"type": "error", "error": "模型回傳但無文字"})
            return

        # 非串流模式：整段清理後直接用原始 on_chunk 分 chunk 送出
        text = _strip_channel_tokens(text)
        if not text:
            filtered_on_chunk({"type": "error", "error": "模型回傳但無文字（過濾後為空）"})
            return

        for i in range(0, len(text), CHUNK_SIZE):
            on_chunk({"type": "delta", "text": text[i: i + CHUNK_SIZE]})
        on_chunk({"type": "done"})
        return
    finally:
        client.close()