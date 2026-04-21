# CS_Agent Backend

FastAPI 後端服務，提供聊天 WebSocket、Guardrail 分類、Ollama 串流整合與健康檢查。

## 需求

- Python 3.10+
- 可用的 Ollama 服務（預設 `http://127.0.0.1:11434`）

## 安裝

```bash
cd backend
python3 -m pip install -r requirements.txt
```

## 啟動

### 開發模式

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 正式模式

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 設定

可透過 `.env` 或系統環境變數設定：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `OLLAMA_MODEL` | `cs-agent-v17` | 預設模型名稱 |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 位址 |
| `OLLAMA_DEBUG` | `false` | 是否開啟除錯日誌 |
| `MAX_MESSAGE_SIZE` | `10240` | WebSocket 單次訊息大小上限（bytes） |
| `HISTORY_MAX_LENGTH` | `20` | 每個 session 保留對話筆數上限 |
| `REQUEST_TIMEOUT` | `30.0` | HTTP 請求超時（秒） |
| `CONNECT_TIMEOUT` | `5.0` | HTTP 連線超時（秒） |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | 允許的來源 |

## API

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

#### 其他控制訊息

- 心跳：`{"type":"ping"}`
- 清除歷史：`{"type":"clear_history"}`
- 結束對話：`{"type":"end_conversation"}`

#### Server 事件

- `delta`：串流片段
- `done`：本次回覆完成
- `error`：錯誤訊息
- `pong`：心跳回覆
- `history_cleared`：歷史已清空
- `idle_warning`：閒置警告
- `conversation_summary`：對話摘要
- `conversation_ended`：對話已結束

## 測試

```bash
cd backend
python3 test.py
```

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
