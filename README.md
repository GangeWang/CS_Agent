# CS_Agent

CS_Agent 是一個以 **FastAPI + React (Vite)** 建置的客服聊天系統，
提供 WebSocket 串流回覆、Guardrail 內容分類、對話歷史管理與對話摘要能力。

## 功能總覽

- 即時 WebSocket 串流聊天（delta / done）
- 客服對話流程（含使用者基本資料輸入）
- Guardrail 分類：`NORMAL` / `ABUSIVE` / `PROMPT_ATTACK` / `SPAM`
- 對話歷史保存與清理（含閒置提醒、結束對話摘要）
- 前端 Markdown 與數學公式（KaTeX）渲染
- 前端內容安全清理（DOMPurify + rehype-sanitize）

## 專案架構

- `backend/`：FastAPI 後端（WebSocket、Guardrail、Ollama 串流）
- `Front/`：React 前端（聊天 UI、Markdown/KaTeX 顯示）

```text
CS_Agent/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/ws.py
│   │   ├── services/
│   │   │   ├── streamer.py
│   │   │   └── guardrail.py
│   │   └── utils/jsonsafe.py
│   ├── backend_ml_ovr_models/
│   ├── classifcation/
│   ├── requirements.txt
│   └── test.py
├── Front/
│   ├── src/
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
└── README.md
```

## 技術棧（完整）

### 前端（Front）

- **React 19**（UI）
- **Vite / rolldown-vite**（前端開發與建置）
- **ESLint 9**（程式碼規範）
- **react-markdown**（Markdown 渲染）
- **remark-gfm / remark-math**（GFM 與數學語法）
- **rehype-katex / KaTeX**（數學公式渲染）
- **rehype-sanitize + hast-util-sanitize**（HTML 白名單清理）
- **DOMPurify**（輸入內容安全清理）
- **rehype-highlight / highlight.js**（程式碼高亮能力）

### 後端服務（backend/app）

- **Python 3.10+**
- **FastAPI**（API / WebSocket）
- **Uvicorn**（ASGI server）
- **httpx**（對 Ollama 的 HTTP 請求）
- **pydantic + pydantic-settings**（設定管理）
- **cachetools (TTLCache)**（對話暫存）
- **websockets**（WebSocket 相關支援）
- **python-multipart**（multipart 支援）

### Guardrail / 機器學習（backend_ml_ovr_models + services/guardrail.py）

- **scikit-learn**（OvR 二元分類模型，TF-IDF + Logistic Regression）
- **joblib**（模型與語料工件存取）
- **NumPy**（向量運算）
- **Sentence Transformers**（語意嵌入比對）
- **規則式分類（Regex Rules）** + **語意比對** + **ML OvR** 三階段判斷

### 訓練腳本（選用）

- `backend/classifcation/*.py`：
  - **pandas**（資料處理）
  - **scikit-learn**（分類訓練/評估）
- `backend/LLM_train.py`：
  - **Unsloth**
  - **Hugging Face datasets**
  - **TRL (SFTTrainer)**
  - **PyTorch**

### 模型服務整合

- **Ollama**（LLM 推論服務）
- 後端使用 `/api/chat` 與 `/api/generate` 進行串流與預熱

## 快速開始

### 1) 啟動後端

```bash
cd /home/runner/work/CS_Agent/CS_Agent/backend
python3 -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) 啟動前端

```bash
cd /home/runner/work/CS_Agent/CS_Agent/Front
npm ci
npm run dev
```

前端預設會連線至：

```text
ws://<目前網頁主機>:8000/ws/chat
```

可透過環境變數覆蓋：

```bash
VITE_WS_URL=ws://your-host:8000/ws/chat
# 或只覆蓋連線埠
VITE_WS_PORT=8000
```

## 後端環境變數

可於 `backend/.env`（參考 `.env.example`）或系統環境變數設定：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `OLLAMA_MODEL` | `cs-agent-v17` | 預設模型名稱 |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 位址 |
| `OLLAMA_DEBUG` | `false` | 是否開啟除錯日誌 |
| `MAX_MESSAGE_SIZE` | `10240` | WebSocket 訊息大小上限（bytes） |
| `HISTORY_MAX_LENGTH` | `20` | 每個 session 保留對話數量上限 |
| `REQUEST_TIMEOUT` | `30.0` | HTTP 請求超時（秒） |
| `CONNECT_TIMEOUT` | `5.0` | HTTP 連線超時（秒） |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | 允許來源 |

## API / WebSocket

### `GET /health`

回傳 API 與 Ollama 狀態。

### `WS /ws/chat`

Client 範例：

```json
{
  "messages": [{"role": "user", "content": "你好"}],
  "model": "cs-agent-v17"
}
```

控制訊息：

- `{"type":"ping"}`
- `{"type":"clear_history"}`
- `{"type":"end_conversation"}`

Server 事件：

- `delta`
- `done`
- `error`
- `pong`
- `history_cleared`
- `idle_warning`
- `conversation_summary`
- `conversation_ended`

## 開發驗證

### 前端

```bash
cd /home/runner/work/CS_Agent/CS_Agent/Front
npm run lint
npm run build
```

### 後端

```bash
cd /home/runner/work/CS_Agent/CS_Agent/backend
python3 test.py
```

> `backend/test.py` 需要可連線的 Ollama（預設 `127.0.0.1:11434`）。

## 授權

目前未提供正式授權聲明，使用前請先與專案擁有者確認。
