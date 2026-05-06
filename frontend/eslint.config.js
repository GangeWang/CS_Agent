import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

// ESLint 組態說明：
// - globalIgnores(['dist'])：忽略建置輸出，避免掃描編譯後檔案。
// - files：只套用在 js/jsx 檔案。
// - extends：沿用 JS 官方建議規則 + React Hooks + Vite React Refresh 建議規則。
// - no-unused-vars：保留既有慣例，允許全大寫或底線開頭名稱（常用於常數/保留符號）。
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs['recommended-latest'],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }],
    },
  },
])
