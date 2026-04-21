# CS_Agent

CS_Agent 是一個以 **FastAPI + React (Vite)** 建置的客服聊天系統，
支援 WebSocket 串流回覆、對話守門分類（Guardrail）、以及前後端分離部署。

## 專案結構

- `backend/`：FastAPI 後端（WebSocket、Guardrail、Ollama 串流）
- `Front/`：React 前端（聊天 UI、Markdown/KaTeX 呈現）

## 快速開始

### 1) 啟動後端

```bash
cd backend
python3 -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) 啟動前端

```bash
cd Front
npm ci
npm run dev
```

預設情況下，前端會連到 `ws://<目前網頁主機>:8000/ws/chat`。
如需覆蓋，請在前端設定環境變數：

```bash
VITE_WS_URL=ws://your-host:8000/ws/chat
```

## 主要功能

- WebSocket 串流回應
- Guardrail 分類（NORMAL / ABUSIVE / PROMPT_ATTACK / SPAM）
- 對話歷史管理與自動清理
- 對話結束摘要事件（`conversation_summary`）
- 前端 Markdown + KaTeX 顯示與內容清理

## 開發與驗證

### 前端

```bash
cd Front
npm run lint
npm run build
```

### 後端

```bash
cd backend
python3 test.py
```

> `backend/test.py` 需要可連線的 Ollama（預設 `127.0.0.1:11434`）。

## 授權

目前未提供正式授權聲明，使用前請先與專案擁有者確認。
