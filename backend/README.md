# CS_Agent Backend

本目錄提供 CS_Agent 的 FastAPI 後端，負責：

- WebSocket 聊天會話（`/ws/chat`）
- Guardrail 文字分類與回覆策略注入
- 串流呼叫 LLM 推理服務
- 健康檢查（`/health`）與啟動預熱

> 本文件不覆蓋 `backend/classifcation/` 的訓練資料與模型細節。

---

## 目錄重點

```text
backend/
├── api/                     # HTTP / WebSocket entry
│   ├── main.py              # FastAPI app、lifespan、/health
│   └── ws.py                # WebSocket route entry
├── orchestrator/            # 主控制層
│   ├── planner.py           # guardrail label -> system instruction
│   ├── executor.py          # WebSocket 對話流程
│   ├── context.py           # payload/history context helpers
│   └── state.py             # orchestration constants
├── tools/                   # Tool facade layer
│   ├── memory.py            # in-process conversation memory
│   ├── llm.py               # LLM streaming facade
│   ├── safety.py            # guardrail facade
│   └── summarize.py         # conversation summary tool
├── services/
│   └── llama_client.py      # LLAMA service HTTP client
├── memory/
│   ├── postgres.py          # persistent memory boundary
│   └── models.py            # memory data models
├── guard/
│   └── policy.py            # guardrail classification policy
├── classifcation/           # 分類資料/訓練相關（另行維護）
├── config.py                # 環境變數設定（Pydantic Settings）
└── requirements.txt
```

---

## 安裝與啟動

請從專案根目錄啟動，讓 `backend.*` 套件匯入路徑保持一致：

```bash
python3 -m pip install -r backend/requirements.txt
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

正式環境範例：

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 設定方式

1. 複製環境檔：

```bash
cp backend/.env.example backend/.env
```

2. 依部署環境調整：

- `LLAMA_API_URL`：LLM 推理服務 URL
- `LLAMA_API_KEY`：若推理服務有做金鑰驗證
- `MAX_MESSAGE_SIZE`：WebSocket 訊息大小上限
- `HISTORY_MAX_LENGTH`：每個 session 的歷史訊息上限
- `CORS_ORIGINS`：允許前端來源

> 實際預設值以 `backend/config.py` 為準。

---

## API 與事件

### `GET /health`

檢查 API 本身與 LLM 服務連線狀態。

### `WS /ws/chat`

前端送出：

```json
{
  "messages": [{"role": "user", "content": "你好"}],
  "model": "your-model"
}
```

控制事件：

- `{"type":"ping"}`
- `{"type":"clear_history"}`
- `{"type":"end_conversation"}`

後端事件：

- `delta`
- `done`
- `error`
- `pong`
- `idle_warning`
- `history_cleared`
- `conversation_summary`
- `conversation_ended`

---

## 開發建議

- `backend/api/` 僅放 HTTP/WebSocket entry，避免把商業流程塞在 router。
- 主要對話流程集中在 `backend/orchestrator/executor.py`。
- 新工具請放在 `backend/tools/`，外部服務 client 請放在 `backend/services/`。
- 若要延伸訊息協定，請同步更新前端 `frontend/src/App.jsx` 的 payload handler。
