# CS_Agent Backend

FastAPI 後端服務，提供聊天 WebSocket、Guardrail 分類、Ollama 串流整合與健康檢查。

## 環境需求

- Python 3.10+
- 可連線 Ollama（預設 `http://127.0.0.1:11434`）

## 安裝

```bash
cd backend
python3 -m pip install -r requirements.txt
```

## 啟動方式

### 開發模式

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 正式模式

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 設定（.env / 系統環境變數）

`.env` 可參考 `.env.example`，實際預設值以 `app/config.py` 為準。

| 變數 | 預設值 | 說明 |
|---|---|---|
| `OLLAMA_MODEL` | `cs-agent-v17` | 預設模型名稱 |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 位址 |
| `OLLAMA_DEBUG` | `false` | 是否開啟除錯日誌 |
| `MAX_MESSAGE_SIZE` | `10240` | WebSocket 單次訊息大小上限（bytes） |
| `HISTORY_MAX_LENGTH` | `20` | 每個 session 保留對話筆數上限 |
| `REQUEST_TIMEOUT` | `30.0` | HTTP 請求超時（秒） |
| `CONNECT_TIMEOUT` | `5.0` | HTTP 連線超時（秒） |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | 允許來源 |

## API / WebSocket

### `GET /health`

回傳 API 與 Ollama 連線狀態。

### `WS /ws/chat`

聊天 WebSocket 端點。

#### Client 訊息格式

```json
{
  "messages": [{"role": "user", "content": "你好"}],
  "model": "cs-agent-v17"
}
```

#### 控制訊息

- `{"type":"ping"}`
- `{"type":"clear_history"}`
- `{"type":"end_conversation"}`

#### Server 事件

- `delta`
- `done`
- `error`
- `pong`
- `history_cleared`
- `idle_warning`
- `conversation_summary`
- `conversation_ended`

## 測試與驗證

```bash
cd backend
python3 test.py
```

> `test.py` 會直接打 Ollama `/api/generate`，若 Ollama 未啟動會出現連線錯誤訊息。

## 目錄

```text
backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── routers/ws.py
│   ├── services/
│   │   ├── streamer.py
│   │   └── guardrail.py
│   └── utils/jsonsafe.py
├── backend_ml_ovr_models/
├── requirements.txt
└── test.py
```
