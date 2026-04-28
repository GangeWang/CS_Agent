# CS_Agent Code Review（2026-04-28）

本次審查聚焦於目前主幹中的 `backend/app` 與 `Front/src/App.jsx`，目標是找出「會影響穩定性、可維護性、與上線風險」的項目。

---

## 總結

- **整體評估：B（可運行，但仍有幾個高優先修正點）**
- **高優先問題（建議先修）**：3 項
- **中優先問題**：4 項
- **低優先與維護建議**：4 項

---

## 高優先問題（High）

### H1. `conversation_summary` 事件未回傳真正摘要內容
- **位置**：`backend/app/routers/ws.py`
- **現象**：`end_conversation()` 已計算 `summary`，但實際送出的 `summary` 固定為「對話已關閉」。
- **影響**：前端拿不到摘要，功能與使用者預期不一致。
- **建議**：將事件 payload 的 `summary` 改為計算結果（例如 `summary` 變數），並保留 fallback 文案。

### H2. 生產預設 LLM URL 為固定外部 IP
- **位置**：`backend/app/config.py`
- **現象**：`llama_api_url` 預設為固定 IP。
- **影響**：開發/測試環境容易誤連生產，且部署可攜性低。
- **建議**：改為本機或空值預設，強制透過 `.env` 注入。

### H3. WebSocket streaming timeout 為硬編碼
- **位置**：`backend/app/routers/ws.py`
- **現象**：`asyncio.wait_for(q.get(), timeout=120)` 寫死。
- **影響**：不同模型速度下不易調整，容易造成誤判超時。
- **建議**：納入 `settings`，與其他 timeout 統一管理。

---

## 中優先問題（Medium）

### M1. 遺留 `print()`，缺少一致 logging
- **位置**：`backend/app/routers/ws.py`
- **建議**：改用 `logger.debug/info`，避免污染 stdout。

### M2. `session_id = id(websocket)` 可讀性與追蹤性一般
- **位置**：`backend/app/routers/ws.py`
- **建議**：改用 `uuid4()` 作為 session id，便於追蹤跨模組日誌。

### M3. 前端重連流程未在頁面離開時取消排程
- **位置**：`Front/src/App.jsx`
- **建議**：保存 reconnect timer，unmount 時清除，避免潛在記憶體洩漏。

### M4. health check 錯誤訊息直接回傳 exception 字串
- **位置**：`backend/app/main.py`
- **建議**：對外回傳通用訊息，詳細錯誤僅記錄到 log。

---

## 低優先與維護建議（Low）

1. 可將 WebSocket payload 型別抽為共享 schema（前後端一致）。
2. `Front/src/App.jsx` 體積偏大，建議拆分 hooks/components。
3. 補上最小整合測試：`/health`、`ws delta/done`、`idle timeout`。
4. 增加部署說明（環境變數、反向代理、CORS 實務）。

---

## 建議修復順序

1. 先修 H1（摘要功能正確性）
2. 再修 H2/H3（配置與 timeout 可控）
3. 接著修 M1/M3（穩定性與維運體驗）
4. 最後進行結構化重構與測試補齊

