# CS_Agent Frontend

React + Vite 前端，提供客服聊天 UI，透過 WebSocket 與 backend 串流互動。

## 環境需求

- Node.js 18+（建議 20+）
- npm 9+
- backend 已啟動且可連線 `ws://<host>:8000/ws/chat`

## 技術棧

- React 19
- Vite（rolldown-vite）
- react-markdown + remark/rehype
- KaTeX + highlight.js
- DOMPurify + rehype-sanitize

## 安裝

```bash
cd Front
npm ci
```

## 開發與建置

```bash
npm run dev
npm run lint
npm run build
npm run preview
```

## WebSocket 設定

預設連線位址：

```text
ws://<目前網頁主機>:8000/ws/chat
```

可用環境變數覆蓋：

```bash
VITE_WS_URL=ws://your-host:8000/ws/chat
# 或只覆蓋連線埠
VITE_WS_PORT=8000
```

## 主要行為

- 串流回覆渲染（`delta` / `done`）
- `ping/pong` 心跳維持連線
- 支援 `idle_warning`、`conversation_summary`、`conversation_ended` 事件
- Markdown / 數學公式 / 程式碼區塊高亮渲染
- 使用者基本資料輸入（姓名、電話）

## 疑難排解

- 畫面無回覆：先檢查 backend 是否啟動於 `8000` 並可接收 `/ws/chat`
- 連線失敗：確認 `VITE_WS_URL` / `VITE_WS_PORT` 是否設定正確
- 建置警告 chunk 過大：屬於目前打包現況，不影響基本功能
