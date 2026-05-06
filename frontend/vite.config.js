import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Vite 設定保持最小化：僅啟用 React plugin，避免引入與本任務無關的建置行為變更。
export default defineConfig({
  plugins: [react()],
})
