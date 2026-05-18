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
- **rehype-highlight + highlight.js**：程式碼區塊語法高亮。
- **rehype-raw + rehype-sanitize**：受控 HTML 解析與安全白名單。
- **DOMPurify**：前端輸入/輸出雙層清理，避免 XSS。

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
npm run build
npm run preview
```

---

## WebSocket 連線設定

預設連線規則：

- 使用當前頁面 host
- 預設 port 為 `8000`
- path 為 `/ws/chat`

可用環境變數覆蓋：

```bash
VITE_WS_URL=ws://your-host:8000/ws/chat
# 或只覆蓋 port
VITE_WS_PORT=8000
```

---

## 開發提醒

- 後端事件型別若變更，需同步調整 `src/App.jsx` 的 `handleWsPayload`。
- 若要優化首屏體驗，可將初始歡迎訊息與個資表單拆成獨立元件。
- lint/build 需在 Node.js 18+ 環境下執行。
