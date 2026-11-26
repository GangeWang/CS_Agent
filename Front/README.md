# CS_Agent 前端

本目錄為 **React + Vite** 前端專案，提供客服聊天 UI，透過 WebSocket 與後端進行即時串流互動。前端重點在於**串流渲染效能、Markdown/數學式顯示、內容安全清理**。

---

## 技術堆疊（詳細）

- **React 19**：UI 組件與狀態更新。
- **Vite（rolldown-vite）**：開發伺服器與正式建置。
- **ESLint**：前端程式碼品質檢查。
- **react-markdown**：Markdown 內容渲染。
- **remark-gfm / remark-math**：GFM 與數學式語法支援。
- **rehype-katex**：KaTeX 數學公式排版。
- **rehype-sanitize**：白名單式 HTML 清理（在 KaTeX 之後執行），避免 XSS；使用者訊息以純文字渲染。

---

## 功能摘要

- **串流接收 `delta`**：採 buffer + flush 方式降低高頻重繪。
- **心跳與重連**：支援 `ping/pong`，斷線後 exponential backoff 重連。
- **對話狀態提示**：`idle_warning`、`conversation_summary`、`conversation_ended`。
- **安全渲染**：Markdown、數學公式、程式碼區塊皆經過清理與白名單控制。

---

## 安裝與執行

```bash
cd Front
npm ci
npm run dev
```

常用指令：

```bash
npm run lint
npm test        # vitest：狀態列邏輯
npm run build
npm run preview
```

---

## WebSocket 連線設定

預設連「同源」的 `/ws/chat`，不需要任何環境變數：

- 開發（`npm run dev`）：`vite.config.js` 把 `/ws/chat` 代理到 `ws://localhost:8000`
- docker-compose：nginx 把 `/ws/chat` 反代到 backend

只有前端與後端不在同一個 host、中間又沒有反代時，才在 `.env` 設定其中一個（見 `.env.example`）：

```bash
VITE_WS_URL=ws://your-host:8000/ws/chat
# 或只指定 port（host 沿用目前頁面）
VITE_WS_PORT=8000
```

`VITE_MODEL_NAME`（可選）：送給後端的模型名稱；後端目前不會轉送給推理服務，沒設定就不送。

---

## 狀態列

每則訊息後依後端回傳的 guardrail label 更新（邏輯在 `src/guardrailUi.js`）：

- `ABUSIVE`（客人有情緒）：「偵測到情緒，已切換安撫模式」，並顯示「情緒強度 x/5」；每則有情緒的訊息 +1，平穩的訊息 -1
- `PROMPT_ATTACK` / `SPAM`：「偵測到高風險請求」，不改變情緒強度
- 其他：「正常」

---

## 開發提醒

- 後端事件型別若變更，需同步調整 `src/App.jsx` 的 `handleWsPayload`。
- lint/build 需在 Node.js 18+ 環境下執行。
