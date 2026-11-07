# Docker 部署指南（後端 + 推理服務）

## 架構

```
docker compose
├─ inference   LLM_gen/server_llama.py + llama-cpp-python（CUDA 編譯，GPU）
│              volume: GGUF 模型檔（不進 image）
├─ backend     backend/app（FastAPI + guardrail，CPU torch）
│              volume: guardrail 模型 / semantic_models / rag_db / HF 快取
└─ (Front 未容器化，開發模式照舊 npm run dev)
```

## 前置需求

- Docker Desktop（WSL2 backend）+ NVIDIA 驅動支援 WSL CUDA
- GPU 驗證：`docker run --rm --gpus all nvidia/cuda:13.4.3-base-ubuntu26.04 nvidia-smi`
- GGUF 模型檔（預設路徑 `./models/CS_AgentV12.gguf`，可用 `LLAMA_MODEL_FILE` 覆蓋）

## 快速開始

```powershell
# 首次建置（inference 的 llama-cpp-python 需現場編譯，約 10-40 分鐘，僅此一次）
docker compose up -d --build

# 確認
docker compose ps
curl http://localhost:8000/health        # {"status":"ok","llama":"connected"}
```

## 常用操作

```powershell
docker compose logs -f inference         # 看模型載入
docker compose logs -f backend
docker compose down                      # 停止（hf-cache volume 保留）
docker compose down -v                   # 連快取一起清
```

## 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `LLAMA_MODEL_FILE` | `./models/CS_AgentV12.gguf` | host 端 GGUF 路徑 |
| `LLAMA_N_CTX`（inference service） | `8192` | 上下文長度 |
| `LLAMA_N_GPU_LAYERS` | `-1` | 全部層進 GPU |
| `GUARDRAIL_DEVICE`（backend service） | `cpu` | 容器內 guardrail 裝置 |

## 設計原則

- **模型檔一律 volume 掛載**，絕不打包進 image（GGUF 數 GB）
- **重訓模型免重建 image**：guardrail 模型、semantic_models、rag_db 都是掛載
- 後端 image 內是 **CPU 版 torch**（體積小）；guardrail 若要容器內跑 CUDA，改 backend/Dockerfile（見檔內註解）
- Session 在進程記憶體中，**勿開多 replica**
