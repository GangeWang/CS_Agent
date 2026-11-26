"""
推理服務回報的模型資訊。

推理服務（LLM_gen/server_llama.py）的 GET /health 會回 {"status", "model", "n_ctx", ...}。
後端把 n_ctx 記下來，讓上下文預算跟著實際載入的模型走，不必手動讓兩邊設定一致：
有效上限 = min(LLAMA_CONTEXT_MAX_TOKENS, 推理服務回報的 n_ctx)。

何時更新：
- 啟動預熱（main.py lifespan）
- 每次有人打後端 /health
- 每輪對話開始前，若還不知道 n_ctx（推理服務比後端晚起來），最多每 30 秒試一次；
  已知時每 10 分鐘重新確認一次（推理服務可能換了設定重啟）
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..config import settings
from .streamer import _get_http_client

logger = logging.getLogger(__name__)

RETRY_UNKNOWN_SECONDS = 30.0
RECHECK_KNOWN_SECONDS = 600.0

_state: dict = {"n_ctx": None, "checked_at": None}


def reported_n_ctx() -> Optional[int]:
    return _state["n_ctx"]


def record_health(body: object) -> None:
    """從推理服務 /health 的 JSON 取出 n_ctx；格式不對就忽略（舊版推理服務沒有這個欄位）。"""
    if not isinstance(body, dict):
        return
    try:
        n_ctx = int(body.get("n_ctx"))
    except (TypeError, ValueError):
        return
    if n_ctx <= 0:
        return
    if n_ctx != _state["n_ctx"]:
        logger.info("Inference service reports n_ctx=%d", n_ctx)
    _state["n_ctx"] = n_ctx


async def probe(timeout: Optional[float] = None) -> str:
    """
    GET 推理服務 /health。回傳 "connected"、"HTTP <code>" 或 "disconnected"，成功時順便記下 n_ctx。
    錯誤細節只進 log。
    """
    url = settings.llama_api_url.rstrip("/") + "/health"
    _state["checked_at"] = time.monotonic()
    try:
        resp = await _get_http_client().get(url, timeout=timeout or settings.request_timeout)
    except Exception as e:
        logger.warning("Inference health check failed: %s", e)
        return "disconnected"
    if resp.status_code != 200:
        logger.warning("Inference health check HTTP %s: %s", resp.status_code, resp.text[:300])
        return f"HTTP {resp.status_code}"
    try:
        record_health(resp.json())
    except Exception:
        pass
    return "connected"


async def refresh_if_stale() -> None:
    """對話前呼叫：只有該重新確認時才真的發請求，失敗不影響對話。"""
    checked_at = _state["checked_at"]
    interval = RECHECK_KNOWN_SECONDS if _state["n_ctx"] else RETRY_UNKNOWN_SECONDS
    if checked_at is not None and time.monotonic() - checked_at < interval:
        return
    await probe(timeout=settings.connect_timeout)


def reset() -> None:
    """測試用。"""
    _state["n_ctx"] = None
    _state["checked_at"] = None
