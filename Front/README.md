# CS_Agent Frontend

本目錄為 React + Vite 前端，提供客服聊天 UI，透過 WebSocket 與後端進行即時串流互動。

---

## 技術堆疊

- React 19
- Vite
- react-markdown + remark-gfm + remark-math
- rehype-katex + rehype-sanitize
- DOMPurify

---

## 安裝與執行

```bash
cd Front
npm ci
npm run dev
```

其他常用指令：

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

## 前端功能摘要

- 串流接收 `delta`，以 buffer + flush 方式降低高頻重繪
- 支援 `ping/pong` 心跳與斷線重連（exponential backoff）
- 顯示 `idle_warning`、`conversation_summary`、`conversation_ended`
- Markdown、數學公式、程式碼內容渲染
- 輸入與渲染雙層內容清理（DOMPurify + rehype-sanitize）

---

## 開發注意

- 後端事件型別若變更，需同步調整 `src/App.jsx` 的 `handleWsPayload`。
- 若要優化首屏體驗，可考慮將初始歡迎訊息與個資表單拆分成獨立元件。
- lint/build 需在 Node.js 18+ 環境下執行。
