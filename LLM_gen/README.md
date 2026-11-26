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
export LLAMA_N_CTX=8192   # 不設時預設 8192；docker-compose 以 LLAMA_N_CTX 統一設定（預設 32768）
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

回報模型是否載入與設定，**不跑推理**（不會跟正常請求搶推理名額）：

```json
{"status": "ok", "model": "CS_AgentV12.gguf", "n_ctx": 32768, "max_concurrent": 1, "busy": false}
```

後端會讀 `n_ctx`，把對話的上下文預算限制在模型實際支援的長度內。
`GET /health?deep=true` 才會實際生成 1 個 token 確認推理正常（忙碌時 15 秒內拿不到名額回 503）。

### `POST /api/generate`

同步回應：

```json
{
  "prompt": "你好",
  "max_tokens": 128
}
```

### `POST /api/stream`

串流回應（SSE）。後端實際送的格式（`messages` 不含 system；`system_prompt` 會放在最前面）：

```json
{
  "system_prompt": "你是客服助理……",
  "messages": [{"role": "user", "content": "你好"}],
  "max_tokens": 512
}
```

回應每段是 `data: {"text": "..."}`，結束是 `data: {"done": true}`，錯誤是 `data: {"error": "generation failed"}`（錯誤細節只進服務端 log）。
沒有 `messages` 時會把 `prompt` 當成單一則 user 訊息。

### 測試

```bash
pip install -r requirements-dev.txt
pytest tests/          # 用假的 llama_cpp 模組，不需要 GPU
```

---

## 與後端整合

請在 `backend/.env` 設定：

```bash
LLAMA_API_URL=http://<llm-server-host>:10000
LLAMA_API_KEY=your-key
```

> 若未使用金鑰驗證，可省略 `LLAMA_API_KEY`。
