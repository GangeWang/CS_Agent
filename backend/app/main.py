"""
Main FastAPI application for CS_Agent backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging

from app.routers.ws import router as ws_router

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

app = FastAPI(title="CS_Agent_Backend_WS")

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
