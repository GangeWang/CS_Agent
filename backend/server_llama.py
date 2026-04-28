from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Any, Optional
import json
import logging
import os
import inspect
import re

try:
    from llama_cpp import Llama
except Exception as e:
    raise RuntimeError("Failed to import llama_cpp binding: " + str(e))

logger = logging.getLogger("server_llama")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Concurrency / executor
MAX_CONCURRENT_INFER = int(os.environ.get("LLAMA_MAX_CONCURRENT", "1"))
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFER)
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_INFER)

# Config via env (defaults)
MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "./CS_AgentV12.gguf")
N_CTX = int(os.environ.get("LLAMA_N_CTX", "2048"))
# DEFAULT GPU LAYERS set to 32 (adjust as needed)
N_GPU_LAYERS = int(os.environ.get("LLAMA_N_GPU_LAYERS", "32"))
N_THREADS = int(os.environ.get("LLAMA_N_THREADS", "4"))
API_KEY = os.environ.get("LLAMA_API_KEY", None)

# instantiate Llama client
try:
    llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, n_threads=N_THREADS)
    logger.info("Llama client created (model_path=%s, n_gpu_layers=%s)", MODEL_PATH, N_GPU_LAYERS)
except Exception as e:
    logger.exception("Failed to create Llama client: %s", e)
    llm = None

def _sanitize_model_output(text: str) -> str:
    """
    清理模型可能帶出的內部標記/思考 (CoT) 等雜訊，嘗試取出最終可見的 assistant 回覆。
    這是 heuristic，若未涵蓋新格式可再調整。
    """
    if not text:
        return text

    # 把 bytes/非 str guard
    if not isinstance(text, str):
        text = str(text)

    # 1) 優先匹配明確的 final 標記 (若模型輸出包含 <|start|>assistant...<|channel|>final<|message|>)
    m = re.search(r"<\|start\|>assistant.*?<\|channel\|>final<\|message\|>(.*)$", text, flags=re.S)
    if m:
        out = m.group(1)
        out = re.sub(r"<\|end\|>.*$", "", out, flags=re.S)
        return out.strip()

    # 2) 移除 analysis channel 區塊
    text = re.sub(r"<\|channel\|>analysis.*?<\|end\|>", "", text, flags=re.S)

    # 3) 移除其他常見內部標記
    text = re.sub(r"<\|start\|>|<\|end\|>|<\|return\|>", "", text, flags=re.S)
    # 移除形如 "<|channel|>...<|message|>" 的內嵌標記
    text = re.sub(r"<\|channel\|>.*?<\|message\|>", "", text, flags=re.S)

    # 4) 常見情況：如果有多段落，最後一段通常為最終回答，取最後非空段落
    parts = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
    if parts:
        return parts[-1]

    return text.strip()

def _extract_text_from_chunk(chunk: Any, system_prompt: Optional[str] = None) -> Optional[str]:
    """
    從各種 chunk 結構抽出 candidate 文字，並做初步過濾（但不做最終 sanitize）。
    """
    if chunk is None:
        return None
    if isinstance(chunk, str):
        txt = chunk.strip()
        if not txt:
            return None
        if system_prompt and txt.startswith(system_prompt.strip()[:200]):
            return None
        return txt

    if isinstance(chunk, dict):
        # chat-style message
        msg = chunk.get("message")
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                if role == "system":
                    return None
                return content

        # choices style
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                # delta style
                delta = first.get("delta")
                if isinstance(delta, dict):
                    cont = delta.get("content") or delta.get("text")
                    if isinstance(cont, str) and cont.strip():
                        if delta.get("role") == "system":
                            return None
                        if system_prompt and cont.strip().startswith(system_prompt.strip()[:200]):
                            return None
                        return cont
                # text in choice
                if isinstance(first.get("text"), str) and first.get("text").strip():
                    txt = first.get("text").strip()
                    if system_prompt and txt.startswith(system_prompt.strip()[:200]):
                        return None
                    return txt

        # other possible keys
        for k in ("text", "response", "response_text", "output", "content"):
            v = chunk.get(k)
            if isinstance(v, str) and v.strip():
                if system_prompt and v.strip().startswith(system_prompt.strip()[:200]):
                    return None
                return v

    return None

# 新增：更健壯的 system_prompt echo 偵測工具
def _normalize_for_compare(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # remove punctuation
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s.lower()

def _looks_like_system_echo(text: str, system_prompt: Optional[str], snippet_len: int = 60) -> bool:
    if not system_prompt:
        return False
    sp_norm = _normalize_for_compare(system_prompt)
    txt_norm = _normalize_for_compare(text)
    if not sp_norm or not txt_norm:
        return False
    sp_snip = sp_norm[:snippet_len]
    # 若輸出以系統片段開頭或包含系統片段，視為回顯
    return txt_norm.startswith(sp_snip) or (sp_snip in txt_norm)

# Flexible invoker: try messages-first, then various prompt/key combos, positional, etc.
def _invoke_func_with_fallback(func, prompt: str, max_tokens: int, stream: bool = True, system_prompt: Optional[str] = None):
    last_exc = None

    # messages-first attempt (preferred for chat-template GGUF)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        return func(messages=messages, max_tokens=max_tokens, stream=stream)
    except TypeError as e:
        last_exc = e
    except Exception:
        raise

    # fallback prompt/text param combos
    prompt_keys = ["prompt", "input", "inputs", "text", "message"]
    token_keys = ["max_tokens", "max_new_tokens", "max_length", "n_predict"]

    for p_key in prompt_keys:
        for t_key in token_keys:
            kwargs = {p_key: prompt, t_key: max_tokens}
            kwargs_stream = kwargs.copy()
            kwargs_stream["stream"] = stream
            try:
                return func(**kwargs_stream)
            except TypeError as e:
                last_exc = e
                try:
                    return func(**kwargs)
                except Exception as e2:
                    last_exc = e2
                    continue
            except Exception:
                raise

    # positional
    try:
        return func(prompt, max_tokens, stream)
    except Exception as e:
        last_exc = e

    # minimal
    try:
        return func(prompt)
    except Exception as e:
        last_exc = e

    raise last_exc or RuntimeError("Failed to invoke model function")

def model_stream_generator(llm_client, prompt: str, max_tokens: int = 256, system_prompt: Optional[str] = None) -> Iterator[Any]:
    if llm_client is None:
        raise RuntimeError("Llama client not initialized")

    candidates = []
    if hasattr(llm_client, "create_chat_completion"):
        candidates.append(llm_client.create_chat_completion)
    if hasattr(llm_client, "create_completion"):
        candidates.append(llm_client.create_completion)
    if hasattr(llm_client, "create"):
        candidates.append(llm_client.create)
    if hasattr(llm_client, "generate"):
        candidates.append(llm_client.generate)
    if callable(llm_client):
        candidates.append(llm_client)

    last_exc = None
    for func in candidates:
        try:
            res = _invoke_func_with_fallback(func, prompt, max_tokens, stream=True, system_prompt=system_prompt)
            if hasattr(res, "__iter__") and not isinstance(res, (str, bytes, dict)):
                for chunk in res:
                    yield chunk
                return
            else:
                yield res
                return
        except Exception as e:
            last_exc = e
            logger.debug("candidate func failed: %s, trying next", e)
            continue

    raise AttributeError("Model client has no supported streaming API (create_chat_completion/create/generate/callable): last error: " + str(last_exc))

def model_generate_once(llm_client, prompt: str, max_tokens: int = 256, system_prompt: Optional[str] = None) -> str:
    if llm_client is None:
        raise RuntimeError("Llama client not initialized")

    preferred = llm_client.create_chat_completion if hasattr(llm_client, "create_chat_completion") else (llm_client.create_completion if hasattr(llm_client, "create_completion") else (llm_client.create if hasattr(llm_client, "create") else (llm_client.generate if hasattr(llm_client, "generate") else llm_client)))
    res = _invoke_func_with_fallback(preferred, prompt, max_tokens, stream=False, system_prompt=system_prompt)

    # merge iterable results
    if hasattr(res, "__iter__") and not isinstance(res, (str, bytes, dict)):
        parts = []
        for chunk in res:
            t = _extract_text_from_chunk(chunk, system_prompt=system_prompt)
            if t:
                parts.append(_sanitize_model_output(t))
            else:
                parts.append(json.dumps(chunk, ensure_ascii=False))
        return "".join(parts)

    # dict: extract best candidate then sanitize
    if isinstance(res, dict):
        # try structured locations
        # 1) chat-style message
        msg = res.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return _sanitize_model_output(content)
        # 2) choices -> message -> content
        choices = res.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            # prefer message.content
            if isinstance(first.get("message"), dict) and isinstance(first["message"].get("content"), str):
                return _sanitize_model_output(first["message"]["content"])
            # fallback to text
            text = first.get("text") or first.get("message", {}).get("content")
            if isinstance(text, str):
                return _sanitize_model_output(text)
        # 3) direct text
        text = res.get("text")
        if isinstance(text, str):
            return _sanitize_model_output(text)
        # fallback: dump whole dict
        return _sanitize_model_output(json.dumps(res, ensure_ascii=False))

    # other types: convert and sanitize
    return _sanitize_model_output(str(res))

def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-KEY")):
    if API_KEY:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@app.get("/health")
async def health():
    if llm is None:
        return JSONResponse(status_code=503, content={"status": "error", "detail": "Llama client not initialized"})
    loop = asyncio.get_running_loop()

    def do_test():
        try:
            for _ in model_stream_generator(llm, prompt="健康檢查", max_tokens=8, system_prompt=None):
                return True
            return False
        except AttributeError:
            try:
                txt = model_generate_once(llm, prompt="健康檢查", max_tokens=8, system_prompt=None)
                return bool(txt)
            except Exception:
                return False
        except Exception:
            return False

    ok = await loop.run_in_executor(executor, do_test)
    return {"status": "ok" if ok else "error"}

@app.post("/api/generate")
async def generate(request: Request, auth=Depends(verify_api_key)):
    data = await request.json()
    max_tokens = int(data.get("max_tokens", 256))
    system_prompt = data.get("system_prompt")
    if "messages" in data:
        msgs = data["messages"]
        # find user content (simple behavior): take last user role content if present
        user_prompt = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break
        prompt = user_prompt or data.get("prompt", "")
    else:
        prompt = data.get("prompt", "")

    loop = asyncio.get_running_loop()

    def blocking():
        try:
            return model_generate_once(llm, prompt=prompt, max_tokens=max_tokens, system_prompt=system_prompt)
        except Exception as e:
            logger.exception("generate failed: %s", e)
            raise

    text = await loop.run_in_executor(executor, blocking)
    return {"text": text}

@app.post("/api/stream")
async def stream(request: Request, auth=Depends(verify_api_key)):
    data = await request.json()
    max_tokens = int(data.get("max_tokens", 256))
    system_prompt = data.get("system_prompt")
    if "messages" in data:
        msgs = data["messages"]
        prompt = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                prompt = m.get("content", "")
                break
    else:
        prompt = data.get("prompt", "")

    loop = asyncio.get_running_loop()

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        def blocking_stream():

            try:
                for chunk in model_stream_generator(llm, prompt=prompt, max_tokens=max_tokens, system_prompt=system_prompt):
                    txt = _extract_text_from_chunk(chunk, system_prompt=system_prompt)
                    if txt:
                        clean = _sanitize_model_output(txt)
                        # 跳過看起來像 system_prompt 回顯的片段
                        if clean and not _looks_like_system_echo(clean, system_prompt):
                            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"text": clean}, ensure_ascii=False))
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"done": True}))
            except AttributeError as ae:
                logger.info("No streaming API; fallback to single generate: %s", ae)
                try:
                    txt = model_generate_once(llm, prompt=prompt, max_tokens=max_tokens, system_prompt=system_prompt)
                    cleaned = _sanitize_model_output(txt)
                    if cleaned and not _looks_like_system_echo(cleaned, system_prompt):
                        loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"text": cleaned}, ensure_ascii=False))
                    loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"done": True}))
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"error": str(e)}))
            except Exception as e:
                logger.exception("streaming failed: %s", e)
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"error": str(e)}))

        await inference_semaphore.acquire()
        try:
            loop.run_in_executor(executor, blocking_stream)
            while True:
                chunk = await queue.get()
                yield f"data: {chunk}\n\n"
        finally:
            inference_semaphore.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")