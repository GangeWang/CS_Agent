# CS_Agent

CS_Agent 是一個以 **FastAPI + React (Vite)** 建置的客服聊天系統，支援 WebSocket 串流、Guardrail 分類、對話歷史管理與結束摘要。

## 主要功能

- 即時串流回覆（`delta` / `done`）
- Guardrail 分類（`NORMAL`、`ABUSIVE`、`PROMPT_ATTACK`、`SPAM`）
- 對話歷史清理、閒置提醒、對話摘要
- 前端 Markdown / KaTeX / 程式碼高亮渲染
- 前端輸入內容安全清理（DOMPurify + sanitize）

## 專案結構

```text
CS_Agent/
├── backend/   # FastAPI + WebSocket + Guardrail + Ollama 串流
├── Front/     # React + Vite 聊天前端
└── README.md
```

詳細說明請看：

- 前端文件：`/home/runner/work/CS_Agent/CS_Agent/Front/README.md`
- 後端文件：`/home/runner/work/CS_Agent/CS_Agent/backend/README.md`

## 快速開始

### 1) 啟動 backend

```bash
cd backend
python3 -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) 啟動 Front

```bash
cd Front
npm ci
npm run dev
```

預設連線：

```text
ws://<目前網頁主機>:8000/ws/chat
```

## 常用驗證指令

```bash
# Front
cd Front && npm run lint && npm run build

# backend（需有可連線 Ollama）
cd backend && python3 test.py
```

## 環境需求

- Node.js 18+（建議 20+）
- Python 3.10+
- 可連線 Ollama 服務（預設 `http://127.0.0.1:11434`）

## 授權

目前尚未提供正式授權聲明，使用前請先與專案擁有者確認。
