# CS_Agent Backend

本目錄提供 CS_Agent 的 FastAPI 後端，負責：

- WebSocket 聊天會話（`/ws/chat`）
- Guardrail 文字分類與回覆策略注入
- 串流呼叫 LLM 推理服務
- 健康檢查（`/health`）與啟動預熱

> 本文件不覆蓋 `backend/classifcation/` 的細節。

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

1. 複製環境檔：

```bash
cp .env.example .env
```

2. 依部署環境調整：

- `LLAMA_API_URL`：LLM 推理服務 URL
- `LLAMA_API_KEY`：若推理服務有做金鑰驗證
- `MAX_MESSAGE_SIZE`：WebSocket 訊息大小上限
- `HISTORY_MAX_LENGTH`：每個 session 的歷史訊息上限
- `CORS_ORIGINS`：允許前端來源

> 實際預設值以 `app/config.py` 為準。

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

- 請先閱讀 `CODE_REVIEW_2026-04-28.md` 再進行重構。
- 若要延伸訊息協定，請同步更新前端 `Front/src/App.jsx` 的 payload handler。
- 若要上線，建議加入：結構化 logging、速率限制、整合測試。
