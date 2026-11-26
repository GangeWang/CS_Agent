"""
CS_Agent 推理服務：llama-cpp-python 的 FastAPI 包裝。

對後端提供的介面（backend/app/services/streamer.py 依賴這個格式）：
- POST /api/stream   SSE：每段 `data: {"text": ...}`，結束 `data: {"done": true}`，錯誤 `data: {"error": ...}`
- POST /api/generate 一次回傳 `{"text": ...}`（後端只在串流還沒送出任何內容就失敗時才用）
- GET  /health       只回報模型是否載入與設定（n_ctx 等），不跑推理；`?deep=true` 才實際生成 1 個 token

請求內容：`messages`（chat 格式，不含 system）、`system_prompt`、`max_tokens`；
沒有 `messages` 時用 `prompt` 當成單一則 user 訊息。

為什麼不直接換成 llama.cpp 內建的 OpenAI 相容 server：後端的請求格式與這裡的輸出清理
（去掉 analysis channel、特殊標記、system prompt 回顯）綁在一起，換掉要同時改後端與部署。
這裡改成直接呼叫 `Llama.create_chat_completion`，拿掉原本「反射式嘗試各種參數組合」的呼叫器
（llama_cpp 用不到，只會把真正的錯誤藏起來）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError("Failed to import llama_cpp binding: " + str(e))

try:
    from llama_cpp import GGML_TYPE_Q8_0
except ImportError:
    from llama_cpp.llama_cpp import GGML_TYPE_Q8_0

logger = logging.getLogger("server_llama")
logging.basicConfig(level=logging.INFO)

app = FastAPI()


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# ── 設定（環境變數） ──────────────────────────────────────────────
MAX_CONCURRENT_INFER = int(os.environ.get("LLAMA_MAX_CONCURRENT", "1"))
MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "./CS_AgentV12.gguf")
# 上下文長度：後端會從 /health 讀這個值，自動把對話預算限制在模型實際支援的範圍內。
# docker-compose 以 LLAMA_N_CTX 同時設定推理服務與後端（預設 32768）；單獨啟動時預設 8192。
N_CTX = int(os.environ.get("LLAMA_N_CTX", "8192"))
N_GPU_LAYERS = int(os.environ.get("LLAMA_N_GPU_LAYERS", "32"))
N_THREADS = int(os.environ.get("LLAMA_N_THREADS", "4"))
API_KEY = os.environ.get("LLAMA_API_KEY") or None
# KV cache 位置：true（預設）= 進 VRAM（快）；false = 留系統 RAM（可開超大 n_ctx，速度較慢）
OFFLOAD_KV = _env_bool("LLAMA_OFFLOAD_KV", "true")
# flash_attn：q8_0 KV 的必要條件；實測對速度幾乎無影響
FLASH_ATTN = _env_bool("LLAMA_FLASH_ATTN", "false")
# KV cache 量化 q8_0：VRAM 佔用減半，實測生成速度損失 ~20%（需 FLASH_ATTN=true）
KV_Q8 = _env_bool("LLAMA_KV_Q8", "false")

# llama_cpp 的 Llama instance 不是執行緒安全的：所有推理都必須過 semaphore，並在單一 worker 執行
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFER)
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_INFER)

try:
    llm: Optional[Llama] = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        n_threads=N_THREADS,
        offload_kqv=OFFLOAD_KV,
        flash_attn=FLASH_ATTN,
        type_k=GGML_TYPE_Q8_0 if KV_Q8 else None,
        type_v=GGML_TYPE_Q8_0 if KV_Q8 else None,
    )
    logger.info(
        "Llama client created (model_path=%s, n_gpu_layers=%s, n_ctx=%d, offload_kqv=%s, flash_attn=%s, kv_q8=%s)",
        MODEL_PATH, N_GPU_LAYERS, N_CTX, OFFLOAD_KV, FLASH_ATTN, KV_Q8,
    )
except Exception as e:
    logger.exception("Failed to create Llama client: %s", e)
    llm = None


# ── 輸出清理 ──────────────────────────────────────────────────────
def _sanitize_model_output(text: str) -> str:
    """
    清理模型可能帶出的內部標記/思考 (CoT) 等雜訊，嘗試取出最終可見的 assistant 回覆。
    優先提取 final channel，嚴格過濾 analysis channel。
    """
    if not text:
        return text
    if not isinstance(text, str):
        text = str(text)

    # 1) 優先匹配明確的 final 標記，這是最可靠的方式
    m = re.search(r"<\|start\|>assistant.*?<\|channel\|>final<\|message\|>(.+?)(?:<\|end\|>|<\|return\|>|$)", text,
                  flags=re.S)
    if m:
        out = m.group(1).strip()
        if out:
            return out

    # 2) 如果沒有 final 標記，完全移除所有 analysis channel
    text = re.sub(r"<\|start\|>assistant.*?<\|channel\|>analysis.*?<\|end\|>", "", text, flags=re.S)

    # 3) 移除其他內部標記
    text = re.sub(r"<\|start\|>|<\|end\|>|<\|return\|>", "", text, flags=re.S)
    text = re.sub(r"<\|channel\|>.*?<\|message\|>", "", text, flags=re.S)

    # 4) 清理空白
    return text.strip()


_BOUNDARY_TOKEN_RE = re.compile(r"<\|start\|>|<\|end\|>|<\|return\|>")


def _clean_stream_chunk(text: str) -> str:
    """
    串流的單一片段只移除邊界標記，**保留空白與換行**。

    原本每個片段都套 _sanitize_model_output（最後會 strip），導致只有換行的片段被丟掉、
    英文單字前的空白被吃掉：Markdown 清單黏成一行、"Hello world" 變成 "Helloworld"。
    analysis / final channel 的切換由後端 streamer 的 _ChannelFilter 跨片段處理。
    """
    return _BOUNDARY_TOKEN_RE.sub("", text)


def _normalize_for_compare(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # remove punctuation
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s.lower()


def _looks_like_system_echo(text: str, system_prompt: Optional[str], snippet_len: int = 60) -> bool:
    """模型把 system prompt 原文吐回來時（輸出以系統片段開頭或包含系統片段），視為回顯並丟棄。"""
    if not system_prompt:
        return False
    sp_norm = _normalize_for_compare(system_prompt)
    txt_norm = _normalize_for_compare(text)
    if not sp_norm or not txt_norm:
        return False
    sp_snip = sp_norm[:snippet_len]
    return txt_norm.startswith(sp_snip) or (sp_snip in txt_norm)


def _chunk_text(chunk: Any) -> Optional[str]:
    """create_chat_completion(stream=True) 的 chunk：choices[0].delta.content。"""
    if not isinstance(chunk, dict):
        return None
    choices = chunk.get("choices")
    if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
        return None
    delta = choices[0].get("delta") or {}
    if delta.get("role") == "system":
        return None
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


def _completion_text(res: Any) -> str:
    """create_chat_completion(stream=False) 的回傳：choices[0].message.content。"""
    try:
        content = res["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"unexpected completion format: {json.dumps(res, ensure_ascii=False)[:300]}")
    return content if isinstance(content, str) else ""


def build_messages(data: dict) -> tuple[list[dict], Optional[str]]:
    """把請求轉成 chat messages；system_prompt 放最前面（messages 自帶 system 時不重複加）。"""
    system_prompt = data.get("system_prompt") or None
    raw = data.get("messages")
    messages: list[dict] = []
    if isinstance(raw, list) and raw:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in raw
            if isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"}
            and isinstance(m.get("content"), str)
        ]
    if not messages:
        messages = [{"role": "user", "content": str(data.get("prompt", ""))}]
    if system_prompt and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages, system_prompt


def _require_llm() -> Llama:
    if llm is None:
        raise RuntimeError("Llama client not initialized")
    return llm


def generate_stream(messages: list[dict], max_tokens: int) -> Iterator[Any]:
    return _require_llm().create_chat_completion(messages=messages, max_tokens=max_tokens, stream=True)


def generate_once(messages: list[dict], max_tokens: int, system_prompt: Optional[str] = None) -> str:
    res = _require_llm().create_chat_completion(messages=messages, max_tokens=max_tokens, stream=False)
    text = _sanitize_model_output(_completion_text(res))
    if text and _looks_like_system_echo(text, system_prompt):
        logger.warning("generate: dropped output that echoes the system prompt")
        return ""
    return text


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-KEY")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def _max_tokens(data: dict) -> int:
    try:
        return max(1, int(data.get("max_tokens", 256)))
    except (TypeError, ValueError):
        return 256


# ── API ──────────────────────────────────────────────────────────
@app.get("/health")
async def health(deep: bool = False):
    """
    預設只回報狀態，不跑推理：原本每次都真的生成，會跟正常請求搶唯一的推理名額，
    而且後端 /health、docker healthcheck 每次輪詢都要等一輪生成。
    deep=true 時才實際生成 1 個 token（忙碌時 15 秒內拿不到名額回 503）。
    """
    info = {
        "model": os.path.basename(MODEL_PATH),
        "n_ctx": N_CTX,
        "max_concurrent": MAX_CONCURRENT_INFER,
        "busy": inference_semaphore.locked(),
    }
    if llm is None:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "Llama client not initialized", **info})
    if not deep:
        return {"status": "ok", **info}

    try:
        await asyncio.wait_for(inference_semaphore.acquire(), timeout=15)
    except asyncio.TimeoutError:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "inference busy", **info})
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, lambda: generate_once([{"role": "user", "content": "健康檢查"}], 1))
    except Exception as e:
        logger.warning("deep health check failed: %s", e)
        return JSONResponse(status_code=503, content={"status": "error", "detail": "inference failed", **info})
    finally:
        inference_semaphore.release()
    return {"status": "ok", "deep": True, **info}


@app.post("/api/generate")
async def generate(request: Request, auth=Depends(verify_api_key)):
    data = await request.json()
    messages, system_prompt = build_messages(data)
    max_tokens = _max_tokens(data)
    loop = asyncio.get_running_loop()

    await inference_semaphore.acquire()
    try:
        text = await loop.run_in_executor(executor, generate_once, messages, max_tokens, system_prompt)
    except Exception as e:
        logger.exception("generate failed: %s", e)
        return JSONResponse(status_code=500, content={"error": "generation failed"})
    finally:
        inference_semaphore.release()
    return {"text": text}


@app.post("/api/stream")
async def stream(request: Request, auth=Depends(verify_api_key)):
    data = await request.json()
    messages, system_prompt = build_messages(data)
    max_tokens = _max_tokens(data)
    loop = asyncio.get_running_loop()

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        # client 斷線（後端逾時取消）或串流結束時 set：生成執行緒在下一個 token 停止，
        # 不再為已離開的請求繼續佔用模型（單一 worker 會擋住後續所有請求）
        stop_event = threading.Event()

        def put(obj: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(obj, ensure_ascii=False))

        def blocking_stream() -> None:
            try:
                gen = generate_stream(messages, max_tokens)
                for chunk in gen:
                    if stop_event.is_set():
                        logger.info("stream cancelled by client; stopping generation")
                        close = getattr(gen, "close", None)
                        if close:
                            close()
                        return
                    txt = _chunk_text(chunk)
                    if not txt:
                        continue
                    clean = _clean_stream_chunk(txt)
                    # 跳過看起來像 system_prompt 回顯的片段
                    if clean and not _looks_like_system_echo(clean, system_prompt):
                        put({"text": clean})
                put({"done": True})
            except Exception as e:
                logger.exception("streaming failed: %s", e)
                put({"error": "generation failed"})

        await inference_semaphore.acquire()
        try:
            loop.run_in_executor(executor, blocking_stream)
            while True:
                raw = await queue.get()
                yield f"data: {raw}\n\n"
                # done/error 後必須結束 generator 釋放 semaphore
                payload = json.loads(raw)
                if payload.get("done") or payload.get("error"):
                    break
        finally:
            stop_event.set()
            inference_semaphore.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
