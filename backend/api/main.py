"""Main FastAPI application for CS_Agent backend."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.ws import router as ws_router
from backend.config import settings
from backend.tools.safety import classify_text

# 後端入口檔（main.py）角色：
# 1) 初始化 FastAPI 應用
# 2) 掛載 CORS 中介層，讓前端可跨網域呼叫
# 3) 提供 /health 健康檢查（含 LLAMA 連線檢測）
# 4) 掛載 WebSocket 路由 /ws/chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GUARDRAIL_WARMUP_TEXT = "系統啟動預熱"


async def _warmup_guardrail_model() -> None:
    """Warm up Guardrail resources without blocking API startup on failure."""
    try:
        await asyncio.to_thread(classify_text, GUARDRAIL_WARMUP_TEXT)
        logger.info("Startup warmup: guardrail resources loaded")
    except Exception as e:
        logger.warning(f"Startup warmup (guardrail) failed: {e}")


async def _warmup_llama_model() -> None:
    """Ping the LLAMA inference service and confirm it is reachable."""
    endpoint = settings.llama_api_url.rstrip("/") + "/health"
    try:
        timeout = httpx.Timeout(
            timeout=settings.request_timeout,
            connect=settings.connect_timeout,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(endpoint)
        if resp.status_code == 200:
            logger.info(f"Startup warmup: LLAMA service at '{settings.llama_api_url}' reachable")
        else:
            logger.warning(f"Startup warmup (llama) failed: HTTP {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.warning(f"Startup warmup (llama) failed: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Service startup lifecycle: warm up guardrail and LLAMA service in parallel."""
    logger.info("Startup warmup: begin")
    await asyncio.gather(
        _warmup_guardrail_model(),
        _warmup_llama_model(),
    )
    logger.info("Startup warmup: end")
    yield


app = FastAPI(title="CS_Agent_Backend_WS", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint for the API and connected LLAMA service."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.llama_api_url.rstrip('/')}/health")
            if response.status_code == 200:
                return {"status": "ok", "llama": "connected"}
            return {"status": "degraded", "llama": "error", "details": f"HTTP {response.status_code}"}
    except Exception as e:
        logger.warning(f"Health check - LLAMA connection failed: {e}")
        return {"status": "degraded", "llama": "disconnected", "error": str(e)}


app.include_router(ws_router)
