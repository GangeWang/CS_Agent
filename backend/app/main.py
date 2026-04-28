"""
Main FastAPI application for CS_Agent backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging
import asyncio
from contextlib import asynccontextmanager

from app.routers.ws import router as ws_router
from app.services.guardrail import classify_text

from app.config import settings

# 後端入口檔（main.py）角色：
# 1) 初始化 FastAPI 應用
# 2) 掛載 CORS 中介層，讓前端可跨網域呼叫
# 3) 提供 /health 健康檢查（含 Ollama 連線檢測）
# 4) 掛載 WebSocket 路由 /ws/chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 啟動預熱時讓模型常駐記憶體的時間；可依部署資源與流量型態調整。
OLLAMA_WARMUP_KEEP_ALIVE = "30m"
OLLAMA_WARMUP_PROMPT = "warmup"
GUARDRAIL_WARMUP_TEXT = "系統啟動預熱"


async def _warmup_guardrail_model() -> None:
    """
    啟動時預熱 Guardrail 模型資源。

    設計目的：
    1) 避免第一位使用者送出訊息時，才觸發模型與語意嵌入器載入，造成首包延遲。
    2) 預熱失敗不應中斷 API 啟動，因此採用「記錄警告 + 持續啟動」策略。
    """
    try:
        # classify_text 內部會透過快取載入模型與語意資源；這裡以背景執行緒先觸發一次。
        await asyncio.to_thread(classify_text, GUARDRAIL_WARMUP_TEXT)
        logger.info("Startup warmup: guardrail resources loaded")
    except Exception as e:
        # 降級策略：即使預熱失敗，也不要讓整體服務啟動失敗。
        logger.warning(f"Startup warmup (guardrail) failed: {e}")


async def _warmup_llama_model() -> None:
    """
    啟動時嘗試 ping WSL 上的 LLAMA 推理服務，確認模型已載入。
    此函式名稱保留以兼容既有程式，但內部改為呼叫 settings.llama_api_url
    """
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
    """
    服務啟動生命週期事件：並行執行模型預熱。

    預熱內容：
    - Guardrail 分類模型與語意資源
    - Ollama 聊天模型

    行為原則：
    - 預熱是最佳化步驟，不是啟動必要條件
    - 任一預熱失敗都不會中止服務，僅記錄日誌供後續排查
    """
    logger.info("Startup warmup: begin")
    await asyncio.gather(
        _warmup_guardrail_model(),
        _warmup_llama_model(),
    )
    logger.info("Startup warmup: end")
    yield


app = FastAPI(title="CS_Agent_Backend_WS", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """
    Health check endpoint that verifies both the API and connected LLAMA (WSL) service.
    Returns:
        dict: Status information including LLAMA connectivity
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 確認 WSL LLAMA server 的 /health
            response = await client.get(f"{settings.llama_api_url.rstrip('/')}/health")
            if response.status_code == 200:
                return {"status": "ok", "llama": "connected"}
            else:
                return {"status": "degraded", "llama": "error", "details": f"HTTP {response.status_code}"}
    except Exception as e:
        logger.warning(f"Health check - LLAMA connection failed: {e}")
        return {"status": "degraded", "llama": "disconnected", "error": str(e)}

# WebSocket 路由
app.include_router(ws_router)
