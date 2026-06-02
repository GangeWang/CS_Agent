# CS_Agent 後端

本目錄提供 CS_Agent 的 **FastAPI 後端**，負責 **WebSocket 聊天會話、Guardrail 文字分類、LLM 串流推理** 與基礎健康檢查。

> 本文件**不覆蓋** `backend/classifcation/` 的細節。

---

## 技術堆疊（詳細）

- **FastAPI**：API/WS 框架與路由管理。
- **Uvicorn（ASGI）**：高效能伺服器，支援 WebSocket。
- **Pydantic / Pydantic Settings**：`.env` 與設定管理。
- **httpx**：呼叫外部 LLM 推理服務（支援串流）。
- **websockets**：WebSocket 通訊支援。
- **cachetools**：快取與 session 狀態管理。
- **python-multipart**：表單資料解析（擴充上傳/表單功能時使用）。

---

## 目錄重點

```text
backend/
├── app/
│   ├── main.py              # FastAPI app、lifespan、/health
│   ├── config.py            # 環境變數設定（Pydantic Settings）
│   ├── routers/ws.py        # WebSocket 主流程與對話狀態
│   └── services/
│       ├── guardrail.py     # 文字分類
│       └── streamer.py      # LLM 串流請求
├── classifcation/           # 分類資料/訓練相關（另行維護）
├── requirements.txt
└── .env.example
```

---

## 安裝與啟動

```bash
cd backend
python3 -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

正式環境範例：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 設定方式

1) 複製環境檔：

```bash
cp .env.example .env
```

2) 依部署環境調整（以 `app/config.py` 為準）：

- `LLAMA_API_URL`：LLM 推理服務 URL
- `LLAMA_API_KEY`：若推理服務需要金鑰
- `MAX_MESSAGE_SIZE`：WebSocket 訊息大小上限
- `HISTORY_MAX_LENGTH`：每個 session 的歷史訊息上限
- `LLAMA_CONTEXT_MAX_TOKENS` / `LLAMA_CONTEXT_RESERVED_OUTPUT_TOKENS`：模型上下文視窗與預留輸出 token；預設參考 Ollama `gpt-oss:20b` 的 128K context window，設定為 `131072` 並預留 `4096` token 給輸出，避免 prompt 撐爆模型限制
- `CONTEXT_STRATEGY`：上下文管理策略，支援 `compress`（預設）或 `window`
- `CONTEXT_COMPRESS_THRESHOLD_RATIO`：預估上下文達到可用預算多少比例時觸發管理（預設 0.8）
- `CONTEXT_COMPRESS_KEEP_RECENT_MESSAGES`：壓縮時保留最近幾則原文訊息
- `CONTEXT_WINDOW_KEEP_RECENT_MESSAGES`：窗口裁切時至少嘗試保留最近幾則訊息
- `CONTEXT_SUMMARY_TARGET_CHARS`：壓縮摘要的目標字數
- `CORS_ORIGINS`：允許前端來源

---


## 上下文長度保護

後端在呼叫 LLM 前會先預估 `system + history + 當前 user` 的 token 數。預設上下文上限以 Ollama `gpt-oss:20b` 的 128K context window（`128 * 1024 = 131072` tokens）為基準，並透過 `LLAMA_CONTEXT_RESERVED_OUTPUT_TOKENS=4096` 預留輸出空間；超過門檻時可用兩種方案處理：

### 方案 1：上下文壓縮（`CONTEXT_STRATEGY=compress`，預設）

- 做法：將較早的對話交給 LLM 摘要成重點，保留最近幾輪原文，並把摘要作為 system context 一起送入下一次推理。
- 優點：能保留長對話的主要需求、已答覆內容、限制條件與下一步，比直接丟棄更不容易失去脈絡。
- 缺點：會多一次 LLM 呼叫，延遲與成本較高；摘要品質取決於模型，仍可能遺漏細節或壓縮錯誤。

### 方案 2：上下文窗口（`CONTEXT_STRATEGY=window`）

- 做法：從最舊訊息開始丟棄，直到剩餘上下文落在模型可用預算內。
- 優點：速度快、成本低、行為可預測，不需要額外 LLM 摘要呼叫。
- 缺點：被丟棄的早期資訊完全消失，長流程客服容易忘記先前條件或承諾。

WebSocket payload 可選擇性帶入 `context_strategy` 覆蓋預設值，例如：

```json
{
  "messages": [{"role": "user", "content": "請延續剛剛的問題"}],
  "model": "your-model",
  "context_strategy": "window"
}
```

---

## API 與事件

### `GET /health`

檢查 API 本身與 LLM 服務連線狀態。

### `WS /ws/chat`

前端送出（客服模式，預設）：

```json
{
  "messages": [{"role": "user", "content": "你好"}],
  "model": "your-model"
}
```

前端送出（Agent 模式）：

```json
{
  "mode": "agent",
  "messages": [{"role": "user", "content": "現在時間"}],
  "model": "your-model"
}
```

控制事件：

- `{"type":"ping"}`
- `{"type":"clear_history"}`
- `{"type":"end_conversation"}`

後端事件（客服模式）：

- `delta`
- `done`
- `error`
- `pong`
- `idle_warning`
- `history_cleared`
- `conversation_summary`
- `conversation_ended`

Agent 模式新增事件：

- `agent_trace`（plan + tool_results）
- `agent_final`

---

## 開發提醒

- 請先閱讀 `CODE_REVIEW_2026-04-28.md` 再進行重構。
- 若要延伸訊息協定，請同步更新前端 `Front/src/App.jsx` 的 payload handler。
- 若要上線，建議加入：結構化 logging、速率限制、整合測試。
