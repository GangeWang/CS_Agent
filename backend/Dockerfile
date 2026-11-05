# CS_Agent 後端（FastAPI + guardrail）
# 預設安裝 CPU 版 torch（guardrail 依 GUARDRAIL_DEVICE 環境變數決定執行裝置）
# 如需容器內 guardrail 走 CUDA，將 base 換為 nvidia/cuda:13.4.3-runtime-ubuntu24.04
# 並把 torch 安裝行改回預設 PyPI index（CUDA 版 wheel 較大）。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf-cache

WORKDIR /app/backend

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# 先裝 CPU 版 torch，再裝其餘依賴（transformers 系偵測到 torch 已存在即不會重裝）
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 應用程式碼；classifcation（guardrail 模型）、semantic_models、app/data（rag_db）
# 由 compose 以 volume 掛載，不打包進 image（重訓模型免重建 image）
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
