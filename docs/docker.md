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
| `GUARDRAIL_FAIL_CLOSED` | 未設（false） | 模型不可用且規則未命中時直接拒答；未設時改用保守指令回覆，不再視為 NORMAL |
| `GUARDRAIL_BLOCK_ANY_URL` | 未設（false） | true 時任何非白名單網址一律判 SPAM（舊行為）；預設只擋多網址或網址 + 引流意圖 |
| `GUARDRAIL_WHITELIST_DOMAINS` | 空 | 額外網址白名單（逗號分隔，含子網域），與內建 `samsung.com` 合併 |

## 設計原則

- **模型檔一律 volume 掛載**，絕不打包進 image（GGUF 數 GB）
- **重訓模型免重建 image**：guardrail 模型、semantic_models、rag_db 都是掛載
- 後端 image 內是 **CPU 版 torch**（體積小）；guardrail 若要容器內跑 CUDA，改 backend/Dockerfile（見檔內註解）
- Session 在進程記憶體中，**勿開多 replica**

## 冒煙測試（不需要 GPU）

把推理服務換成假模型（真的 `server_llama.py` + 假的 `llama_cpp`），其餘照正式設定跑：前端 nginx、後端、倉庫裡真的 guardrail 模型（要先 `git lfs pull`）。適合改完程式後確認整套能起來、串流與心跳正常，再上 GPU。

```bash
docker compose -f docker-compose.yml -f docker/smoke/compose.smoke.yml up -d --build
pip install websockets
python docker/smoke/ws_smoke.py http://<這台機器的 IP>:8080   # 結束代碼 0 = 通過
docker compose -f docker-compose.yml -f docker/smoke/compose.smoke.yml down
```

- 用 IP 或網域，不要用 localhost（localhost 在後端 Origin 白名單裡，測不出 nginx 有沒有保留埠號）。
- 檢查項目：`/health` 為 ok、每句都有 guardrail 標籤、回覆逐段送出、回覆途中的 ping 有 pong、思考內容沒有外洩。
- 假模型預設模仿 llama-cpp-python 不印特殊標記的輸出（`analysis…assistantfinal…`）；`FAKE_MODE=harmony` 改成帶 `<|channel|>` 標記的輸出。
- 換成真的推理服務（不加 `-f docker/smoke/compose.smoke.yml`）也能跑 `ws_smoke.py`，可當上 GPU 後的驗收。
