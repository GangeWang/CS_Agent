import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './style.css' // 如果 styles 在 src，或改成 '/styles.css' 指向根目錄

// 前端應用進入點：
// 1) 取得 index.html 中 id=root 的容器
// 2) 以 React 19 的 createRoot 啟動 App
// 3) 使用 StrictMode 協助開發階段提早發現副作用與潛在問題
ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
)
