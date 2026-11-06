import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Vite 設定保持最小化：僅啟用 React plugin 與開發代理。
// 開發模式走同源 /ws/chat，由 vite proxy 轉發到本機後端，
// 與容器化部署（nginx 反代）行為一致。
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ws/chat': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
