# LLM_gen（本地 LLM 推理服務）

本目錄提供一個以 **FastAPI + llama.cpp** 包裝的本地推理服務，供後端以 HTTP 呼叫。

---

## 功能摘要

- 提供 `/api/generate` 與 `/api/stream` 介面
- 以 SSE 方式串流回覆（`data: {...}`）
- 可選 X-API-KEY 驗證
- 可透過環境變數設定模型路徑與推理參數

---

## 安裝與啟動

1. 安裝依賴（需具備 llama.cpp Python binding）：

```bash
cd LLM_gen
python3 -m pip install -r requirements.txt
# 依平台安裝 llama-cpp-python（需支援本機編譯或 CUDA）
python3 -m pip install llama-cpp-python
```

2. 設定必要環境變數（至少需要模型路徑）：

```bash
export LLAMA_MODEL_PATH=./CS_AgentV12.gguf
export LLAMA_N_CTX=2048
export LLAMA_N_GPU_LAYERS=32
export LLAMA_N_THREADS=4
export LLAMA_MAX_CONCURRENT=1
# 若需 API key
export LLAMA_API_KEY=your-key
```

3. 啟動服務（可自行調整 port）：

```bash
uvicorn server_llama:app --host 0.0.0.0 --port 10000
```

---

## API 介面

### `GET /health`

簡單健康檢查，會嘗試觸發一次極短推理。

### `POST /api/generate`

同步回應：

```json
{
  "prompt": "你好",
  "max_tokens": 128
}
```

### `POST /api/stream`

串流回應（SSE）：

```json
{
  "prompt": "你好",
  "max_tokens": 128
}
```

---

## 與後端整合

請在 `backend/.env` 設定：

```bash
LLAMA_API_URL=http://<llm-server-host>:10000
LLAMA_API_KEY=your-key
```

> 若未使用金鑰驗證，可省略 `LLAMA_API_KEY`。
