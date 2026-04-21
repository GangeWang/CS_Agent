# CS_Agent Frontend

React + Vite 前端，提供客服聊天 UI，透過 WebSocket 與後端串流互動。

## 技術

- React 19
- Vite
- react-markdown + remark/rehype
- KaTeX

## 安裝

```bash
cd Front
npm ci
```

## 啟動

```bash
npm run dev
```

## 建置

```bash
npm run build
```

## Lint

```bash
npm run lint
```

## WebSocket 設定

預設會連線到：

```text
ws://<目前網頁主機>:8000/ws/chat
```

若要指定其他位址，設定：

```bash
VITE_WS_URL=ws://your-host:8000/ws/chat
# 或只覆蓋連線埠
VITE_WS_PORT=8000
```

## 功能摘要

- 串流回覆顯示（delta/done）
- 心跳保活與自動重連
- 對話結束與摘要事件處理
- Markdown 與數學公式渲染
- 基本使用者資料輸入（姓名/電話）

## 常用指令

```bash
npm run dev
npm run lint
npm run build
npm run preview
```
