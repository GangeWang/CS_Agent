"""
Main FastAPI application for CS_Agent backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from contextlib import asynccontextmanager

from app.routers.ws import router as ws_router
from app.services import inference_info
from app.services.context_manager import effective_context_max_tokens
from app.services.guardrail import classify_text, guardrail_status
from app.services.streamer import aclose_http_client

from app.config import settings

# 後端入口檔（main.py）角色：
# 1) 初始化 FastAPI 應用
# 2) 掛載 CORS 中介層，讓前端可跨網域呼叫
# 3) 提供 /health 健康檢查（含推理服務與 guardrail 狀態）
# 4) 掛載 WebSocket 路由 /ws/chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """啟動時 ping 推理服務 /health：確認模型已載入，並記下它回報的 n_ctx（決定上下文預算）。"""
    status = await inference_info.probe()
    if status == "connected":
        logger.info(
            "Startup warmup: LLAMA service at '%s' reachable, context budget %d tokens",
            settings.llama_api_url, effective_context_max_tokens(),
        )
    else:
        logger.warning("Startup warmup (llama) failed: %s", status)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    服務啟動生命週期事件：並行執行模型預熱。

    預熱內容：
    - Guardrail 分類模型與語意資源
    - LLM 推理服務連線檢查

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
    # 關閉 streamer 共用的 AsyncClient 連線池
    await aclose_http_client()


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
    Health check endpoint：確認 API 本體、LLM 推理服務與 guardrail 模型狀態。

    錯誤細節只進 log，回應不攜帶 exception 字串（避免內部資訊外洩）。
    Returns:
        dict: status / llama / guardrail / context_max_tokens（實際使用的上下文上限）
    """
    gr = guardrail_status()
    llama = await inference_info.probe(timeout=5.0)
    llama_ok = llama == "connected"
    return {
        "status": "ok" if (llama_ok and gr["loaded"]) else "degraded",
        "llama": llama,
        "guardrail": "ok" if gr["loaded"] else "degraded",
        "context_max_tokens": effective_context_max_tokens(),
    }

# WebSocket 路由
app.include_router(ws_router)
logger.info("LLM inference URL: %s", settings.llama_api_url)