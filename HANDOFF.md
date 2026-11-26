# CS_Agent 交接文件（2026-11-26）

> 這份文件記錄 2026-11-26 這一輪「架構技術債分析與修復」的完整狀態，讓下一位接手的人（或下一個 Claude session）不用重新問一遍就能繼續。
> 所有變更都在分支 `claude/project-thread-nvor7j`，對應 [PR #35](https://github.com/GangeWang/CS_Agent/pull/35)，**尚未 merge**。

---

## 1. 一句話現況

後端的高優先 bug 已修好；guardrail 訓練資料已清理並補上人工樣本；六支訓練腳本已改為共用切分與評估。
**但所有模型都還沒有用新資料重新訓練**：上線的 guardrail 仍是舊資料訓練的版本，下一步要在有 GPU 的機器上重訓。

---

## 2. 與專案負責人確認過的決策（不要再重問）

| 主題 | 決策 |
|---|---|
| Guardrail 的目的 | 判斷使用者**有沒有情緒**，有情緒就採**安撫**策略 |
| `ABUSIVE` 標籤定義 | ＝「有情緒」：不滿、焦急、失望、憤怒都算，**有沒有髒話都算**。例：「客服裝死不理人！準備接客訴」是 ABUSIVE |
| `NORMAL` 標籤定義 | 語氣平穩的一般詢問，即使內容有「立即、投資、垃圾、加入」等字眼 |
| `PROMPT_ATTACK` / `SPAM` | 拒答 |
| 架構選擇 | 負責人的實測結果是 **OvR 比 4選1 好**，上線維持 OvR。新的統一評估可用來再確認（見第 6 節） |
| 補充樣本的方式 | **人工撰寫**，不要用本機 ollama 擴增（品質不可靠）；ollama 擴增腳本已移除 |
| 資料庫 | **還沒搭建**。對話狀態存在記憶體、agent 長期記憶是 stub，都是刻意的，等 DB 建好再處理 |

---

## 3. 本次完成的工作（PR #35 的 commit）

| Commit | 內容 |
|---|---|
| `126d896` fix | 高優先技術債：guardrail 規則誤判、串流中途斷線重複輸出、逾時無法取消生成、guardrail 降級時等同放行 |
| `7ebdaca` data | 語料審查與清理（`step09_clean_dataset.py`），產出 `train_full_dataset_v2.csv` |
| `8207713` data | 依「ABUSIVE = 有情緒」定義修正人工種子 |
| `42d85b6` data | 手寫補齊 537 筆優質樣本，並收緊「垃圾」辱罵規則 |
| `fdf961c` refactor | 六種分類方式共用切分與評估，修正輸出路徑與方法問題 |
| `0e6d92c` fix | 語意庫改用 v2 資料，修正互動測試工具的預設路徑 |

### 3.1 後端修復（`backend/app/`）
- **Guardrail 規則**（`services/guardrail.py`）：「立即、投資、入會、禮品卡」等單一常用詞，改成必須同時帶非白名單網址才算 SPAM。「垃圾」只在罵人語境才命中。網址白名單支援子網域，並能擋下偽裝網域。單一網址不再直接判為 SPAM。
- **串流**（`services/streamer.py`）：已送出內容後才中斷的話，不再 fallback 重新生成，避免重複輸出。新增 `cancel_event`。
- **WebSocket**（`routers/ws.py`）：抽出 `_pump_stream`，chat 和 agent 共用。逾時或斷線時會取消背景生成，也不再卡住連線。
- **降級**：模型不可用而且規則未命中時，改用保守的 `UNVERIFIED` 指令，不再當成 NORMAL。
- **推理服務**（`LLM_gen/server_llama.py`）：client 斷線時停止生成。

### 3.2 訓練資料（`backend/classifcation/`）
- 舊資料的主要問題（完整內容見 `classifcation/README.md` 的「語料審查與清理」一節）：
  - 592 筆小寫 `spam` 被訓練腳本整批丟掉。
  - ABUSIVE 有 90% 是英文維基討論頁的翻譯。
  - PROMPT_ATTACK 有 1,553 筆是缺少圖片的多模態題。
  - 翻譯失敗殘留英文 1,282 筆。
  - NORMAL 全是 26 字以內的模板句，模型會學到長度捷徑。
  - 大量近重複資料。
- `data_set/train_full_dataset_v2.csv`：10,533 筆（NORMAL 4,691 / ABUSIVE 3,639 / SPAM 1,458 / PROMPT_ATTACK 745）。
- `data_set/removed_rows_v2.csv`：每一筆被移除的資料和原因。
- `data_pipeline/curated_seed_data.py`、`curated_seed_extra.py`：537 筆人工樣本。
- `data_set/hard_eval.csv`：47 筆困難評估集，**不進訓練**。重訓後最重要的就是看這份的成績。

### 3.3 訓練腳本（`backend/classifcation/`）
- 新增 `common.py`，六支腳本共用同一份資料、同一個切分（seed 42）、同一套評估，輸出路徑也固定下來。
- **最嚴重的 bug**：輸出路徑原本是相對 CWD，重訓後模型寫不到上線讀取的位置，上線一直是舊模型，而且不會報錯。
- OvR 改為以上線決策規則組合成 4 類結果再評估，可以直接和 4選1 比較。
- 門檻在 val 上挑選，平手時取平台中位數（舊版的 4 個門檻全是搜尋下限 0.30，代表沒有真正選過）。
- BiLSTM 改用 `pack_padded_sequence`，所有方法都加上 class weight。
- 新增 `evaluate_guardrail_pipeline.py`（端到端評估上線流程）、`compare_reports.py`（六種方法比較表）。
- 可以用 `GUARDRAIL_BASE_MODEL` 切換預訓練模型做 A/B 測試，輸出到另一個目錄，不會覆蓋上線模型。

---

## 4. 下一步（依優先順序）

1. **Review 並 merge PR #35**。
2. **在有 GPU 的機器上重訓並評估**（指令見第 5 節）：
   - 跑 `run_all_training.ps1`，跑完用 `compare_reports.py` 看六種方法在 `hard_eval` 上的成績。
   - 用數據確認 OvR 和 4選1 哪個好。4選1 的推論成本只有 OvR 的 1/4，如果兩者差不多，值得考慮。
3. **A/B 測試預訓練模型**：`ckiplab/bert-base-chinese`（繁體中文語料預訓練）對上現在的 `hfl/chinese-roberta-wwm-ext`（以簡體語料為主）。
4. **重建語意庫**：`python backend/classifcation/build_semantic_models.py`（已改用 v2）。
5. **重訓後跑端到端評估**：`python backend/classifcation/evaluate_guardrail_pipeline.py`，確認上線流程（規則 → ML → 語意）的整體表現。
6. 視評估結果決定是否**再補 NORMAL 長句**（目前 30 字以上只有 116 筆），由人工撰寫。
7. 處理第 8 節剩下的技術債。

---

## 5. 怎麼重訓與驗證

```powershell
cd backend\classifcation
.\run_all_training.ps1                       # 六種方法 → reports/*.json → 上線流程評估 → 比較表
python compare_reports.py                    # 只看比較表

# 換預訓練模型做 A/B（輸出到 ..._models_clean__ckiplab-bert-base-chinese，不覆蓋上線模型）
$env:GUARDRAIL_BASE_MODEL="ckiplab/bert-base-chinese"; python transformer\train_ovr_transformer.py

python build_semantic_models.py              # 重訓上線模型後重建語意庫
python evaluate_guardrail_pipeline.py        # 端到端評估
```

後端測試：

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/          # 目前 51 個，全部通過
```

> 另外還有 `backend/tests_smoke.py`（較早的整合煙霧測試，會起一支 uvicorn 走完 WebSocket 流程，`python tests_smoke.py` 直接執行），跟 `tests/` 是兩套東西，這次沒有改它。

---

## 6. Sandbox 參考數據（沒有 GPU，只跑了 ML 和 BiLSTM）

用完整的 v2 資料和共用切分實際訓練。test 有 2,107 筆，hard_eval 有 47 筆。**hard_eval 樣本少，只能看方向，不能下定論。**

| 方法 | test macro-F1 | test 準確率 | hard_eval macro-F1 | hard_eval 準確率 | hard_eval UNCERTAIN |
|---|---|---|---|---|---|
| TF-IDF + LR 4選1 | 0.973 | 0.979 | 0.809 | 0.830（錯 8 筆） | 0% |
| TF-IDF + LR OvR | 0.966 | 0.969（UNCERTAIN 2.1%） | 0.735 | 0.660（錯 16 筆） | 21.3% |
| BiLSTM 4選1 | 0.958 | 0.972 | 0.761 | 0.766（錯 11 筆） | 0% |
| BiLSTM OvR | 0.966 | 0.969（UNCERTAIN 2.0%） | **0.847** | **0.851**（錯 7 筆） | 0% |
| Transformer 4選1 / OvR | 需要 GPU，尚未跑 | | | | |

OvR 在 val 上選出的門檻：
- TF-IDF：NORMAL 0.45 / ABUSIVE 0.69 / PROMPT_ATTACK 0.65 / SPAM 0.55
- BiLSTM：NORMAL 0.34 / ABUSIVE 0.81 / PROMPT_ATTACK 0.92 / SPAM 0.78

觀察：
- **test 分數都在 96～98%，hard_eval 卻只有 66～85%**。這代表 test 仍然偏簡單（模板化資料），真正的難點在 hard_eval 這類真實語句。之後判斷模型好壞時，請以 hard_eval 為主。
- **OvR 和 4選1 哪個好，在不同模型上結論不一樣**：
  - BiLSTM 上 OvR 明顯較好（0.847 對 0.761），這和負責人「OvR 較好」的實測方向一致。
  - TF-IDF 上反而是 4選1 較好，因為 OvR 有 21% 的 hard_eval 樣本沒有任何類別過門檻，落入 UNCERTAIN。上線時 UNCERTAIN 會再交給語意比對，所以這不等於上線的錯誤率。
  - hard_eval 只有 47 筆，一筆約 2%。BiLSTM OvR 和 TF-IDF 4選1 只差 1 筆，分不出高下。
  - Transformer 的結果要等 GPU 重訓後才知道，請用新的統一評估再確認一次。
- 常見錯誤（各方法的錯誤明細在 `reports/<方法>.json` 的 `errors`）：
  - **英文訊息**：正常的英文詢問被判成 SPAM 或 ABUSIVE，英文的 prompt attack 被判成 ABUSIVE。v2 裡以英文為主的句子只有 38 筆（約 0.4%），模型等於沒學過英文；推測英文字母多的句子容易被當成網址或廣告。
  - **含官方網址的正常詢問**被判成 SPAM（例：附上 samsung.com 訂單連結）。規則層的白名單只會讓規則不命中，模型本身仍然可能判成 SPAM。
  - **冒充主管或老闆的越權請求**被 TF-IDF 和 BiLSTM 4選1 判成 ABUSIVE。
  - 「系統提示一直出現錯誤代碼 502」這類含「系統提示」字眼的正常詢問，被 BiLSTM 判成 PROMPT_ATTACK。

---

## 7. 已知限制與風險

- **所有模型都還沒用 v2 重訓**，上線 guardrail 的行為目前只有規則部分改變了。
- `server_llama.py` 的改動只做過語法檢查，沒有在 GPU 環境實際跑過。
- 後端測試在 sandbox 是用 stub 取代 torch/transformers 跑的，這些測試本身不載入模型。
- NORMAL 長句仍然偏少，長度捷徑只是減輕，還沒有完全消除。
- 英文樣本很少（以英文為主的只有 38 筆），英文訊息容易誤判。如果客群會用英文，需要補英文樣本。
- `GUARDRAIL_FAIL_CLOSED` 預設仍是關閉，模型缺檔時會用保守指令回覆，而不是拒答。

---

## 8. 剩餘技術債（尚未處理）

來自第一輪架構審查，依優先順序排列：

| # | 項目 | 備註 |
|---|---|---|
| 6 | 對話狀態綁在單一 WS 連線上，斷線就消失，無法水平擴展 | **等 DB 建好再處理** |
| 7 | 後端全部用同步 httpx 再丟進 thread，上下文壓縮和結束摘要也會同步呼叫 LLM | 可改為 async httpx 串流 |
| 8 | 上下文長度有三套預設值（backend 128K、compose 32K、推理服務 2048），再加上 `history_max_length=20` 的硬截斷，compress 策略幾乎不會觸發 | 統一成單一設定來源 |
| 9 | `server_llama.py` 有一套反射式的「嘗試各種參數組合」呼叫器；`/health` 每次都真的跑推理 | 可改用 llama.cpp 內建的 OpenAI 相容 server |
| 10 | guardrail 模組一被 import 就有副作用；每則訊息依序跑 4 個 transformer，沒有 batch | 如果改成 4選1，這項會一併解決 |
| 11 | RAG 命中門檻偏低（2-gram 重疊 3 個就算），命中後 prompt 會強制「以此為核心回答」 | `rag_db.json` 的內容是聯通電信 FAQ，跟其他資料的產業不一致，也要確認 |
| 12 | 沒有 CI；前端沒有測試 | 建議先加 GitHub Actions 跑 `pytest tests/` |
| 13 | `classifcation/` 裡有約 100MB 的原始 CSV 和中間產物；目錄名 `classifcation` 拼錯；`Front/style.txt` 殘留；`backend/expan.py`、`backend/test.py` 是已失效的 Ollama 時代腳本 | 改目錄名會牽動很多路徑，要一次改乾淨 |
| 14 | 前端：模型名稱寫死；收集了姓名電話卻沒用到；`.env.example` 的 `VITE_WS_PORT=8000` 會繞過 nginx 同源反代；`ws.py` 的 `ws_chat` 仍是巨型函式 | |
| — | 前端的 guardrail 文案是「偵測到不當語氣」，跟「有情緒、需安撫」的定義不完全一致 | 跟負責人確認是否要改文案 |

---

## 9. 重要檔案地圖

```
backend/
├── app/routers/ws.py                 WebSocket 主流程（guardrail → RAG → context → 串流）
├── app/services/guardrail.py         規則 + OvR transformer + 語意比對；MODEL_DIR = 上線模型位置
├── app/services/streamer.py          呼叫推理服務、串流過濾、cancel_event
├── tests/test_smoke.py               後端回歸測試
├── tests/test_training_common.py     訓練共用模組測試
└── classifcation/
    ├── README.md                     模型結果、語料審查結論、訓練與評估說明
    ├── common.py                     ★ 六種方法共用：資料、切分、門檻、決策、報告、輸出路徑
    ├── run_all_training.ps1          一鍵重訓 + 評估 + 比較
    ├── evaluate_guardrail_pipeline.py 端到端評估上線流程
    ├── compare_reports.py            六種方法比較表
    ├── build_semantic_models.py      重建語意庫（v2）
    ├── data_pipeline/step09_clean_dataset.py   清理 + 合併人工樣本 → v2
    ├── data_pipeline/curated_seed_*.py         人工樣本（標籤定義寫在檔頭）
    ├── data_set/train_full_dataset_v2.csv      訓練資料
    ├── data_set/hard_eval.csv                  困難評估集（不進訓練）
    ├── data_set/removed_rows_v2.csv            被移除的資料與原因
    ├── ML/ BILSTM/ transformer/                各方法訓練腳本（+ bilstm_lib.py、hf_lib.py）
    └── transformer/backend_ml_ovr_transformer_models_clean/   🏆 上線模型
```

## 10. 新增或變更的環境變數

| 變數 | 預設 | 用途 |
|---|---|---|
| `GUARDRAIL_WHITELIST_DOMAINS` | 空 | 額外的網址白名單（逗號分隔，含子網域） |
| `GUARDRAIL_BLOCK_ANY_URL` | false | true 時恢復舊行為：任何非白名單網址一律判為 SPAM |
| `GUARDRAIL_FAIL_CLOSED` | false | 模型不可用而且規則未命中時直接拒答 |
| `GUARDRAIL_BASE_MODEL` | `hfl/chinese-roberta-wwm-ext` | 訓練時切換預訓練模型（A/B 用，不會覆蓋上線模型） |
