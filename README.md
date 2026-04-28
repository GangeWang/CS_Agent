# CS_Agent

CS_Agent 是一個以 **FastAPI（後端）+ React/Vite（前端）** 建置的即時客服對話系統，具備：

- WebSocket 串流回覆（`delta` / `done`）
- Guardrail 分類（一般、辱罵、提示攻擊、垃圾訊息）
- 對話歷史管理、閒置警告與結束流程
- 前端 Markdown / KaTeX / 程式碼區塊渲染與內容清理

---

## 專案結構

```text
CS_Agent/
├── backend/                 # FastAPI + WebSocket + Guardrail + LLM 串流
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/ws.py
│   │   └── services/
│   ├── classifcation/       # 分類模型相關（本 README 不展開）
│   └── README.md
├── Front/                   # React + Vite 聊天前端
│   └── README.md
└── README.md
```

---

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

預設前端會連到：

```text
ws://<目前網頁主機>:8000/ws/chat
```

---

## 環境需求

- Python 3.10+
- Node.js 18+（建議 20+）
- npm 9+
- 可連線的 LLM 推理服務（依 `backend/app/config.py` 設定）

---

## 文件導覽

- 後端說明：`backend/README.md`
- 前端說明：`Front/README.md`
- 本次程式審查：`CODE_REVIEW_2026-04-28.md`

---

## 注意事項

- `backend/classifcation/` 為獨立分類資料與模型目錄，請依該目錄內文件與流程管理。
- 正式環境請務必設定 `.env`，不要使用預設 URL / timeout 直接上線。
