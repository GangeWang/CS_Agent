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

# Add CORS middleware
async def _warmup_guardrail_model() -> None:
    """
    啟動時預熱 Guardrail 模型資源。

    設計目的：
    1) 避免第一位使用者送出訊息時，才觸發模型與語意嵌入器載入，造成首包延遲。
    2) 預熱失敗不應中斷 API 啟動，因此採用「記錄警告 + 持續啟動」策略。
    """
    try:
        # classify_text 內部會透過快取載入模型與語意資源；這裡以背景執行緒先觸發一次。
        await asyncio.to_thread(classify_text, "系統啟動預熱")
        logger.info("Startup warmup: guardrail resources loaded")
    except Exception as e:
        # 降級策略：即使預熱失敗，也不要讓整體服務啟動失敗。
        logger.warning(f"Startup warmup (guardrail) failed: {e}")


async def _warmup_ollama_model() -> None:
    """
    啟動時預熱 Ollama LLM 模型。

    設計目的：
    1) 讓模型在服務啟動階段先進入記憶體，避免第一個聊天請求等待模型冷啟動。
    2) 只做最小化請求（空 prompt + 非串流），降低預熱成本。

    注意：
    - 若 Ollama 當下不可用，僅記錄 warning，不阻塞後端服務啟動。
    """
    endpoint = settings.ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": "",
        "stream": False,
        "keep_alive": "30m",
    }

    try:
        timeout = httpx.Timeout(
            timeout=settings.request_timeout,
            connect=settings.connect_timeout,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json=payload)

        if resp.status_code == 200:
            logger.info(f"Startup warmup: Ollama model '{settings.ollama_model}' loaded")
        else:
            logger.warning(f"Startup warmup (ollama) failed: HTTP {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.warning(f"Startup warmup (ollama) failed: {e}")


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
        _warmup_ollama_model(),
        return_exceptions=True,
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
    Health check endpoint that verifies both the API and Ollama connection.
    
    Returns:
        dict: Status information including Ollama connectivity
    """
    try:
        # Check Ollama connection
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
            if response.status_code == 200:
                # API 與 Ollama 都可用
                return {"status": "ok", "ollama": "connected"}
            else:
                # API 正常，但 Ollama 回傳非 200，標記為 degraded 便於監控告警
                return {"status": "degraded", "ollama": "error", "details": f"HTTP {response.status_code}"}
    except Exception as e:
        # 連線失敗時仍回應健康資訊，避免監控端直接超時
        logger.warning(f"Health check - Ollama connection failed: {e}")
        return {"status": "degraded", "ollama": "disconnected", "error": str(e)}


# WebSocket 路由
app.include_router(ws_router)
