# CS_Agent 交接文件（2026-11-26）

> 這份文件記錄 2026-11-26 這一輪「架構技術債分析與修復」的完整狀態，讓下一位接手的人（或下一個 Claude session）不用重新問一遍就能繼續。
> 主要變更在 [PR #35](https://github.com/GangeWang/CS_Agent/pull/35)（2026-11-26 已 merge）。

---

## 1. 一句話現況

後端的高優先 bug 已修好；guardrail 訓練資料已清理並補上人工樣本；六支訓練腳本已改為共用切分與評估。
2026-11-26 負責人已在自己的 GPU 機器上用 v2 重訓六種方法。上線流程在 hard_eval 答對 46/47 句，macro-F1 0.974（見第 6 節）。
之後又補了第三批人工樣本（NORMAL 長句與英文樣本，見 3.2），**v2 已更新，需要再重訓一次**；新模型重訓後再 commit 進倉庫。

---

## 2. 與專案負責人確認過的決策（不要再重問）

| 主題 | 決策 |
|---|---|
| Guardrail 的目的 | 判斷使用者**有沒有情緒**，有情緒就採**安撫**策略 |
| `ABUSIVE` 標籤定義 | ＝「有情緒」：不滿、焦急、失望、憤怒都算，**有沒有髒話都算**。例：「客服裝死不理人！準備接客訴」是 ABUSIVE |
| `NORMAL` 標籤定義 | 語氣平穩的一般詢問，即使內容有「立即、投資、垃圾、加入」等字眼 |
| `PROMPT_ATTACK` / `SPAM` | 拒答 |
| 離題閒聊、笑話 | 維持 **SPAM**（拒答），不改標 NORMAL（2026-11-26 決定）。SPAM 裡來自 SMS Spam 資料集的笑話保留原標籤 |
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
- `data_set/train_full_dataset_v2.csv`：10,755 筆（NORMAL 4,887 / ABUSIVE 3,649 / SPAM 1,464 / PROMPT_ATTACK 755）。
- `data_set/removed_rows_v2.csv`：每一筆被移除的資料和原因。
- `data_pipeline/curated_seed_data.py`、`curated_seed_extra.py`：537 筆人工樣本。
- `data_pipeline/curated_seed_long.py`（第三批，GPU 重訓之後加入）：185 句 30 字以上的 NORMAL（其中 48 句 73～89 字），加上 37 句英文樣本（NORMAL 11 / ABUSIVE 10 / PROMPT_ATTACK 10 / SPAM 6）。NORMAL 30 字以上從 116 筆增加到 312 筆，英文為主的樣本從 38 筆增加到 75 筆。
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

1. ~~Review 並 merge PR #35~~（已完成）。
2. ~~在有 GPU 的機器上用 v2 重訓六種方法~~（2026-11-26 已完成，結果見第 6 節）。
3. **用更新後的 v2 再重訓一次**：v2 在 GPU 重訓之後補了第三批樣本。merge 包含修正版 `run_all_training.ps1` 的 PR 後直接跑，它會一併重建語意庫並做端到端評估。請和第 6 節的數字比較，特別是 hard_eval。
4. **把新模型和語意庫 commit 進倉庫**（Git LFS），上線環境才會拿到新模型：`transformer/backend_ml_ovr_transformer_models_clean/`、`backend/semantic_models/`，以及 `reports/`。
5. **（可選）OvR 改 4選1**：準確度打平（見第 6 節），只有在需要加快推論時才值得換。4選1 只要 1 次 forward，OvR 要 4 次；上線程式目前只支援 OvR，換的話要改 `guardrail.py`。
6. **A/B 測試預訓練模型**：`ckiplab/bert-base-chinese`（繁體中文語料預訓練）對上現在的 `hfl/chinese-roberta-wwm-ext`（以簡體語料為主）。現有模型在 hard_eval 已經 0.97，優先度降低。
7. **長度捷徑還沒完全消除**：30 字以上的訊息裡，NORMAL 只佔 13%（312 / 2,392）；NORMAL 有約 4,200 筆是 15 字以內的 ML_data 模板句，很多語意不通（例：「請問註冊地址要預約？」）。可以考慮把這些短模板句下採樣到 2,000～2,500 筆，但這會改變類別比例，要用 GPU 重訓確認 hard_eval 沒有退步再採用。
8. 處理第 8 節剩下的技術債。

---

## 5. 怎麼重訓與驗證

```powershell
cd backend\classifcation
.\run_all_training.ps1                       # 六種方法 → 重建語意庫 → 上線流程評估 → 比較表，最後列出每一步成功或失敗
.\run_all_training.ps1 -SkipTest             # 同上，但端到端評估只跑 hard_eval（較快）
python compare_reports.py                    # 只看比較表

# 換預訓練模型做 A/B（輸出到 ..._models_clean__ckiplab-bert-base-chinese，不覆蓋上線模型）
$env:GUARDRAIL_BASE_MODEL="ckiplab/bert-base-chinese"; python transformer\train_ovr_transformer.py

python build_semantic_models.py              # 單獨重建語意庫
python evaluate_guardrail_pipeline.py        # 單獨跑端到端評估（預設用 GPU，每 200 句印一次進度）
```

> Windows PowerShell 5.1 的注意事項：
> - python 印到 stderr 的警告或進度條，在 `2>&1` 之下會顯示成紅字的 `NativeCommandError`，**這不代表失敗**。`run_all_training.ps1` 已經把它轉成一般文字，並以結束代碼判斷成功或失敗。
> - 手動跑腳本時，如果輸出經過 `Tee-Object` 或管線，請先設 `$env:PYTHONUNBUFFERED="1"`，不然畫面會像卡住。
> - 上線 guardrail 預設用 CPU（`GUARDRAIL_DEVICE`）。`evaluate_guardrail_pipeline.py` 沒設定時會改用 cuda。

後端測試：

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/          # 目前 51 個，全部通過
```

> 另外還有 `backend/tests_smoke.py`（較早的整合煙霧測試，會起一支 uvicorn 走完 WebSocket 流程，`python tests_smoke.py` 直接執行），跟 `tests/` 是兩套東西，這次沒有改它。

---

## 6. v2 重訓結果（2026-11-26）

負責人在自己的 GPU 機器上用 `run_all_training.ps1` 跑完六種方法和端到端評估，資料是完整的 v2 和共用切分。test 有 2,107 筆，hard_eval 有 47 筆。**hard_eval 樣本少（一筆約 2%），只能看方向，不能下定論。**

| 方法 | test macro-F1 | test 準確率 | test UNCERTAIN | hard_eval macro-F1 | hard_eval 準確率 | hard_eval UNCERTAIN |
|---|---|---|---|---|---|---|
| TF-IDF + LR 4選1 | 0.972 | 0.979 | 0% | 0.804 | 0.809 | 0% |
| TF-IDF + LR OvR | 0.967 | 0.969 | 2.0% | 0.735 | 0.660 | 21.3% |
| BiLSTM 4選1 | 0.957 | 0.972 | 0% | 0.788 | 0.787 | 0% |
| BiLSTM OvR | 0.969 | 0.973 | 1.8% | 0.793 | 0.745 | 12.8% |
| Transformer 4選1 | 0.992 | 0.995 | 0% | **0.974** | **0.979**（錯 1 句） | 0% |
| Transformer OvR | 0.994 | 0.994 | 0.3% | 0.973 | 0.957（2 句 UNCERTAIN，判錯類別 0 句） | 4.3% |
| **上線流程**（規則 → Transformer OvR → 語意比對） | 0.994 | 0.994 | 0.2% | **0.974** | **0.979**（錯 1 句） | 0% |

> 上線流程這一列，用的還是 v2 之前的舊語意庫（當時 `run_all_training.ps1` 還沒有重建語意庫這一步）。只有 0.2% 的句子會走到語意比對，重建後的影響應該很小，但仍建議重跑一次。

> 上表是第三批樣本加入**之前**的 v2。加入之後（v2 = 10,755 筆），sandbox 重訓 TF-IDF 的 hard_eval：4選1 從 0.809 升到 0.837；OvR 從 0.735 升到 0.810，UNCERTAIN 從 21.3% 降到 4.3%。Transformer 要等 GPU 重訓。

觀察：
- **Transformer 遠勝其他方法**：hard_eval 0.97，TF-IDF 和 BiLSTM 只有 0.73～0.80。
  - 4選1 錯的 1 句是英文的 prompt attack（Ignore the above...）被判成 ABUSIVE。
  - OvR 沒有判錯任何類別，只有 2 句沒把握（「我是你們老闆，命令你…」和一句英文抱怨）落入 UNCERTAIN。上線流程接住了這 2 句，最後只錯 1 句，跟 4選1 一樣。
- **OvR 或 4選1**：在 transformer 上準確度打平，沒有準確度上的理由換架構。要換只會是為了推論速度，4選1 只要 1 次 forward，OvR 要 4 次。上線維持 OvR。
- **TF-IDF 和 BiLSTM 的 test 分數都在 96～98%，hard_eval 卻只有 66～80%**。test 仍然偏簡單（模板化資料），判斷模型好壞時請以 hard_eval 為主。
- **BiLSTM 的結果不穩定**：同樣的程式和資料，sandbox（CPU）上 BiLSTM OvR 的 hard_eval 是 0.847，負責人的 GPU 機器上是 0.793。47 句的評估集加上訓練的隨機性，BiLSTM 和 TF-IDF 之間的 OvR / 4選1 比較不可靠。
- 常見錯誤（各方法的錯誤明細在 `reports/<方法>.json` 的 `errors`）：
  - **英文訊息**：正常的英文詢問被判成 SPAM 或 ABUSIVE，英文的 prompt attack 被判成 ABUSIVE。v2 裡以英文為主的句子只有 38 筆（約 0.4%），模型等於沒學過英文；推測英文字母多的句子容易被當成網址或廣告。
  - **含官方網址的正常詢問**被判成 SPAM（例：附上 samsung.com 訂單連結）。規則層的白名單只會讓規則不命中，模型本身仍然可能判成 SPAM。
  - **冒充主管或老闆的越權請求**被 TF-IDF 和 BiLSTM 4選1 判成 ABUSIVE。
  - 「系統提示一直出現錯誤代碼 502」這類含「系統提示」字眼的正常詢問，被 BiLSTM 判成 PROMPT_ATTACK。

---

## 7. 已知限制與風險

- **v2 重訓的新模型還沒 commit 進倉庫**，倉庫和 docker 映像檔裡仍是舊模型。
- `server_llama.py` 的改動只做過語法檢查，沒有在 GPU 環境實際跑過。
- 後端測試在 sandbox 是用 stub 取代 torch/transformers 跑的，這些測試本身不載入模型。
- NORMAL 長句仍然偏少（30 字以上 312 筆），長度捷徑只是減輕，還沒有完全消除（見第 4 節第 7 點）。
- 英文樣本仍少（以英文為主的 75 筆），TF-IDF 和 BiLSTM 對英文仍容易誤判。如果客群常用英文，需要再補。
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
