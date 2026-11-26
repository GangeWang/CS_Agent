# CS_Agent 後端

本目錄提供 CS_Agent 的 **FastAPI 後端**，負責 **WebSocket 聊天會話、Guardrail 文字分類、LLM 串流推理** 與基礎健康檢查。

> 本文件**不覆蓋** `backend/classifcation/` 的細節。

---

## 技術堆疊（詳細）

- **FastAPI**：API/WS 框架與路由管理。
- **Uvicorn（ASGI）**：高效能伺服器，支援 WebSocket。
- **Pydantic / Pydantic Settings**：`.env` 與設定管理。
- **httpx**：呼叫外部 LLM 推理服務（支援串流）。
- **websockets**：WebSocket 通訊支援。
- **cachetools**：快取與 session 狀態管理。
- **python-multipart**：表單資料解析（擴充上傳/表單功能時使用）。

---

## 目錄重點

```text
backend/
├── app/
│   ├── main.py              # FastAPI app、lifespan、/health
│   ├── config.py            # 環境變數設定（Pydantic Settings）
│   ├── routers/ws.py        # WebSocket 主流程與對話狀態
│   └── services/
│       ├── guardrail.py     # 文字分類
│       └── streamer.py      # LLM 串流請求
├── classifcation/           # 分類資料/訓練相關（另行維護）
├── requirements.txt
└── .env.example
```

---

## 安裝與啟動

```bash
cd backend
python3 -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

正式環境範例：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 設定方式

1) 複製環境檔：

```bash
cp .env.example .env
```

2) 依部署環境調整（以 `app/config.py` 為準）：

- `LLAMA_API_URL`：LLM 推理服務 URL
- `LLAMA_API_KEY`：若推理服務需要金鑰
- `MAX_MESSAGE_SIZE`：WebSocket 訊息大小上限
- `HISTORY_MAX_LENGTH`：每個 session 在記憶體裡最多保留的訊息數（預設 100，只是防記憶體爆掉的硬上限；送給模型多少歷史由下面的 token 預算決定）
- `LLAMA_CONTEXT_MAX_TOKENS` / `LLAMA_CONTEXT_RESERVED_OUTPUT_TOKENS`：上下文上限與預留給輸出的 token（預設 `32768` / `4096`）。後端會讀推理服務 `/health` 回報的 `n_ctx`，實際上限取兩者較小值，所以通常不用手動對齊
- `CONTEXT_STRATEGY`：上下文管理策略，支援 `compress`（預設）或 `window`
- `CONTEXT_COMPRESS_THRESHOLD_RATIO`：預估上下文達到可用預算多少比例時觸發管理（預設 0.8）
- `CONTEXT_COMPRESS_KEEP_RECENT_MESSAGES`：壓縮時保留最近幾則原文訊息
- `CONTEXT_WINDOW_KEEP_RECENT_MESSAGES`：窗口裁切時至少嘗試保留最近幾則訊息
- `CONTEXT_SUMMARY_TARGET_CHARS`：壓縮摘要的目標字數
- `CORS_ORIGINS`：允許前端來源

---


## 上下文長度保護

後端在呼叫 LLM 前會先預估 `system + history + 當前 user` 的 token 數。上下文上限 = `min(LLAMA_CONTEXT_MAX_TOKENS, 推理服務回報的 n_ctx)`：

- 推理服務的 `n_ctx` 在啟動預熱、每次 `/health`、以及對話時（尚未取得時每 30 秒、已取得時每 10 分鐘）更新
- 後端 `/health` 會回 `context_max_tokens`，可以直接看目前實際用的上限
- docker-compose 只要設一個 `LLAMA_N_CTX`（預設 32768），推理服務與後端都吃這個值

再扣掉 `LLAMA_CONTEXT_RESERVED_OUTPUT_TOKENS`（最多扣一半）就是可用預算；超過門檻（預設 80%）時可用兩種方案處理：

### 方案 1：上下文壓縮（`CONTEXT_STRATEGY=compress`，預設）

- 做法：將較早的對話交給 LLM 摘要成重點，保留最近幾輪原文，並把摘要作為 system context 一起送入下一次推理。
- 優點：能保留長對話的主要需求、已答覆內容、限制條件與下一步，比直接丟棄更不容易失去脈絡。
- 缺點：會多一次 LLM 呼叫，延遲與成本較高；摘要品質取決於模型，仍可能遺漏細節或壓縮錯誤。

### 方案 2：上下文窗口（`CONTEXT_STRATEGY=window`）

- 做法：從最舊訊息開始丟棄，直到剩餘上下文落在模型可用預算內。
- 優點：速度快、成本低、行為可預測，不需要額外 LLM 摘要呼叫。
- 缺點：被丟棄的早期資訊完全消失，長流程客服容易忘記先前條件或承諾。

WebSocket payload 可選擇性帶入 `context_strategy` 覆蓋預設值，例如：

```json
{
  "messages": [{"role": "user", "content": "請延續剛剛的問題"}],
  "model": "your-model",
  "context_strategy": "window"
}
```

---

## FAQ 檢索（RAG）

知識庫是 `app/data/rag_db.json`（`[{"question", "aliases", "answer"}]`），檔案改了會自動重新載入。

- 比對：完全相同直接命中；否則把「請問、怎麼、可以」這類贅詞去掉，中文取雙字詞、英數取單字，
  用 IDF 加權算相似度（0～1），達到 `RAG_MIN_SCORE`（預設 0.4）才附給模型。
- 命中時，提示會請模型先判斷這筆 FAQ 是否真的回答了問題，不相關就不要硬套。
- guardrail 判成 `PROMPT_ATTACK` / `SPAM` 的訊息一律拒答，不查 RAG。
- 換 FAQ 或調門檻後跑 `python tools/eval_rag.py`（可加 `--min-score 0.5`）：
  會列出換句話問法的命中率（`tools/rag_eval_queries.csv`，換 FAQ 時要一起換）與 hard_eval 199 句的誤命中。

目前數字（舊做法「雙字詞重疊數 ≥ 3」→ 新做法）：換句話問法 31/37 → 37/37 命中正確；
hard_eval 誤命中 52/199（含 13 句提示詞攻擊）→ 3/199（其中 2 句是同主題不同題目）。

---

## API 與事件

### `GET /health`

回 `status`（`ok` / `degraded`）、`llama`（推理服務：`connected` / `HTTP <code>` / `disconnected`）、
`guardrail`（模型是否已載入）、`context_max_tokens`（目前實際使用的上下文上限）。

### `WS /ws/chat`

前端送出（客服模式，預設）：

```json
{
  "messages": [{"role": "user", "content": "你好"}],
  "model": "your-model"
}
```

前端送出（Agent 模式）：

```json
{
  "mode": "agent",
  "messages": [{"role": "user", "content": "現在時間"}],
  "model": "your-model"
}
```

控制事件：

- `{"type":"ping"}`
- `{"type":"clear_history"}`
- `{"type":"end_conversation"}`

後端事件（客服模式與 Agent 模式相同）：

- `guardrail`（`label`、`degraded`；前端只用來更新狀態列）
- `delta`
- `done`
- `error`
- `pong`
- `idle_warning`
- `history_cleared`
- `conversation_summary`
- `conversation_ended`

Agent 模式的 plan 與工具結果只寫進後端 log，不送到前端。

---

## 開發提醒

- 交接與目前狀態見倉庫根目錄 `HANDOFF.md`；舊的審查文件在 `docs/archive/`。
- 若要延伸訊息協定，請同步更新前端 `Front/src/App.jsx` 的 payload handler。
- 測試：`pip install -r requirements.txt -r requirements-dev.txt && pytest -q tests/`（不需要真的模型或推理服務）。
