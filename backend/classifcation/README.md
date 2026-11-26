# Guardrail 模型結果整理



## BiLSTM（多類）

```text
=== Classification Report ===
               precision    recall  f1-score   support

      ABUSIVE     0.9091    0.9091    0.9091        33
       NORMAL     0.9961    0.9971    0.9966      1028
PROMPT_ATTACK     0.9565    0.8800    0.9167        25
         SPAM     0.9434    0.9615    0.9524        52

     accuracy                         0.9903      1138
    macro avg     0.9513    0.9369    0.9437      1138
 weighted avg     0.9903    0.9903    0.9903      1138
```

---
## BiLSTM OvR

```text
Training binary model: NORMAL vs NOT_NORMAL
[NORMAL] threshold(from val) = 0.64
              precision    recall  f1-score   support

           0     0.9727    0.9727    0.9727       110
           1     0.9971    0.9971    0.9971      1028

    accuracy                         0.9947      1138
   macro avg     0.9849    0.9849    0.9849      1138
weighted avg     0.9947    0.9947    0.9947      1138

Training binary model: ABUSIVE vs NOT_ABUSIVE
[ABUSIVE] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     0.9982    0.9991    0.9986      1105
           1     0.9688    0.9394    0.9538        33

    accuracy                         0.9974      1138
   macro avg     0.9835    0.9692    0.9762      1138
weighted avg     0.9973    0.9974    0.9973      1138


Training binary model: PROMPT_ATTACK vs NOT_PROMPT_ATTACK
[PROMPT_ATTACK] threshold(from val) = 0.48
              precision    recall  f1-score   support

           0     1.0000    0.9991    0.9996      1113
           1     0.9615    1.0000    0.9804        25

    accuracy                         0.9991      1138
   macro avg     0.9808    0.9996    0.9900      1138
weighted avg     0.9992    0.9991    0.9991      1138


Training binary model: SPAM vs NOT_SPAM
[SPAM] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     0.9972    0.9991    0.9982      1086
           1     0.9800    0.9423    0.9608        52

    accuracy                         0.9965      1138
   macro avg     0.9886    0.9707    0.9795      1138
weighted avg     0.9965    0.9965    0.9965      1138
```

---
## TF-IDF + LR（多類）

```text
=== Classification Report ===
               precision    recall  f1-score   support

      ABUSIVE     0.8889    0.9697    0.9275        33
       NORMAL     0.9990    0.9961    0.9976      1028
PROMPT_ATTACK     0.9615    1.0000    0.9804        25
         SPAM     0.9804    0.9615    0.9709        52

     accuracy                         0.9938      1138
    macro avg     0.9575    0.9818    0.9691      1138
 weighted avg     0.9942    0.9938    0.9939      1138
```

---

## TF-IDF + LR（OvR）

```text
Training binary model: NORMAL vs NOT_NORMAL
[NORMAL] best threshold = 0.32
              precision    recall  f1-score   support

           0     0.9818    0.9818    0.9818       110
           1     0.9981    0.9981    0.9981      1028

    accuracy                         0.9965      1138
   macro avg     0.9899    0.9899    0.9899      1138
weighted avg     0.9965    0.9965    0.9965      1138

Training binary model: ABUSIVE vs NOT_ABUSIVE
[ABUSIVE] best threshold = 0.82
              precision    recall  f1-score   support

           0     0.9973    0.9991    0.9982      1105
           1     0.9677    0.9091    0.9375        33

    accuracy                         0.9965      1138
   macro avg     0.9825    0.9541    0.9678      1138
weighted avg     0.9964    0.9965    0.9964      1138

Training binary model: PROMPT_ATTACK vs NOT_PROMPT_ATTACK
[PROMPT_ATTACK] best threshold = 0.66
              precision    recall  f1-score   support

           0     0.9991    1.0000    0.9996      1113
           1     1.0000    0.9600    0.9796        25

    accuracy                         0.9991      1138
   macro avg     0.9996    0.9800    0.9896      1138
weighted avg     0.9991    0.9991    0.9991      1138

Training binary model: SPAM vs NOT_SPAM
[SPAM] best threshold = 0.52
              precision    recall  f1-score   support

           0     0.9982    0.9991    0.9986      1086
           1     0.9804    0.9615    0.9709        52

    accuracy                         0.9974      1138
   macro avg     0.9893    0.9803    0.9847      1138
weighted avg     0.9973    0.9974    0.9974      1138

```

---
## Transformer（多類）

```text
=== Classification Report ===
               precision    recall  f1-score   support

      ABUSIVE     0.9706    1.0000    0.9851        33
       NORMAL     0.9990    0.9990    0.9990      1028
PROMPT_ATTACK     1.0000    0.9600    0.9796        25
         SPAM     1.0000    1.0000    1.0000        52

     accuracy                         0.9982      1138
    macro avg     0.9924    0.9898    0.9909      1138
 weighted avg     0.9983    0.9982    0.9982      1138
```

---

## Transformer（OvR）

```text
Training binary model: NORMAL vs NOT_NORMAL
[NORMAL] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000       110
           1     1.0000    1.0000    1.0000      1028

    accuracy                         1.0000      1138
   macro avg     1.0000    1.0000    1.0000      1138
weighted avg     1.0000    1.0000    1.0000      1138


Training binary model: ABUSIVE vs NOT_ABUSIVE
[ABUSIVE] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     1.0000    0.9991    0.9995      1105
           1     0.9706    1.0000    0.9851        33

    accuracy                         0.9991      1138
   macro avg     0.9853    0.9995    0.9923      1138
weighted avg     0.9991    0.9991    0.9991      1138

Training binary model: PROMPT_ATTACK vs NOT_PROMPT_ATTACK
[PROMPT_ATTACK] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     0.9991    1.0000    0.9996      1113
           1     1.0000    0.9600    0.9796        25

    accuracy                         0.9991      1138
   macro avg     0.9996    0.9800    0.9896      1138
weighted avg     0.9991    0.9991    0.9991      1138

Training binary model: SPAM vs NOT_SPAM
[SPAM] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000      1086
           1     1.0000    1.0000    1.0000        52

    accuracy                         1.0000      1138
   macro avg     1.0000    1.0000    1.0000      1138
weighted avg     1.0000    1.0000    1.0000      1138
```

---

## 資料集溯源與管線

最終訓練集 `data_set/train_full_dataset.csv`（21,567 筆）由下列管線產生（腳本於 `data_pipeline/`，依步驟編號）：

```
【外部原始】                        【抽取 step01-03】            【翻譯 step06 (ollama)】
HF parquet (32.4MB) ──→ extracted_abusive.csv ──→ abusive_translated_zh.csv (9,433)
sms-spam-collection.csv ──→ extracted_spam.csv ──→ spam_translated_zh.csv (747)
JailBreakV_28K.csv ──→ extracted_prompt_attack.csv ─step04 去重→ deduped_prompt_attack.csv (4,172) ──→ prompt_attack_translated_zh.csv (4,172)

【合併 step07】08-21
    三個翻譯檔 + data_set/ML_data.csv（7 月舊世代 18,689 筆）
  = merged_full_dataset.csv（33,041 筆）

【去重 step08】08-21
    merged_full_dataset.csv → deduped_full_dataset.csv（21,567 筆）
  = data_set/train_full_dataset.csv（md5 與去重產物一致）

【消費】run_all_training.ps1 → 六種分類方式訓練腳本
```

- 歷史檔名對照：`full_data_set_fillted.csv`（filtered 錯字）= 現 `train_full_dataset.csv`；`FILLTTER/` = 現 `data_pipeline/`
- 合併與翻譯步驟原為手動/遺失腳本，step06/07 為事後補寫使管線可完整重現
- 已知資料特性：去重後仍有 592 筆 label 為小寫 `spam`（其餘為大寫 `SPAM`），訓練/評估時注意
- 完整實驗結果與選型理由見本檔上方各方法 Classification Report

---

## 六種分類方式矩陣（3 演算法 × 2 架構）

| | 4選1（單模型多分類） | OVR（4 個二元分類器，每 label 一個模型） |
|---|---|---|
| **ML**（TF-IDF + LR） | `ML/train_multiclass_ml.py` → `backend_ml_guardrail_model.joblib` | `ML/train_ovr_ml.py` → `ML/backend_ml_ovr_models/`（4 個 _bin.joblib） |
| **BiLSTM**（字元級，packed） | `BILSTM/train_multiclass_bilstm.py` → `backend_ml_guardrail_bilstm_pytorch.pt` + vocab + label_encoder | `BILSTM/train_ovr_bilstm.py` → `BILSTM/backend_ml_ovr_bilstm_models/`（4 .pt + vocab） |
| **Transformer**（hfl/chinese-roberta-wwm-ext） | `transformer/train_multiclass_transformer_legacy.py` → `transformer/backend_ml_guardrail_transformer/` | `transformer/train_ovr_transformer.py` → `transformer/backend_ml_ovr_transformer_models_clean/` 🏆 **上線模型** |

### 訓練與評估（2026-11-26 起）

六支腳本共用 `common.py`：同一份資料（`data_set/train_full_dataset_v2.csv`）、同一個切分（train 72% / val 8% / test 20%，seed 42，近重複已去除）、
同一套評估，輸出路徑固定（不受執行目錄影響）。每支腳本都會：

1. 在 **val** 上選門檻（OvR；F1 平台取中位數，平台過寬會印 `[WARN]`）
2. 在 **test** 上報告；OvR 以上線決策規則（過門檻取最高、都沒過 → UNCERTAIN）組合成 4 類結果，與 4選1 可直接比較
3. 在 **`data_set/hard_eval.csv`**（困難樣本，不進訓練）上報告並列出錯誤樣本
4. 寫入 `reports/<method>.json`

```powershell
.\run_all_training.ps1                       # 六種方法 + 上線流程評估 + 比較表
python compare_reports.py                    # 只看比較表
python evaluate_guardrail_pipeline.py        # 評估上線完整流程（RULE → ML → SEMANTIC）
$env:GUARDRAIL_BASE_MODEL="ckiplab/bert-base-chinese"; python transformer\train_ovr_transformer.py  # 換預訓練模型 A/B，輸出到另一個目錄，不覆蓋上線模型
```

重訓上線模型後，記得一併重建語意庫：`python build_semantic_models.py`（已改用 v2 資料）。

> 命名對照歷史：原 `BILSTM.py` = multiclass（非 binary）；原 `transformer.py` = multiclass 實驗版；原 `binary_transformer.py` = 上線 OVR 訓練腳本（「binary」指每個二元分類器）。

### 模型 binary 入庫政策（2026-11-07 更新）

六套模型的 binary **全部以 Git LFS 入庫**（含非上線五套，供對照與回滾）；
僅 `transformer/backend_ml_guardrail_transformer/checkpoints/`（訓練中間態）排除。

---

## 語料審查與清理（2026-11-26，step09）

審查 `train_full_dataset.csv`（21,567 筆）發現的主要髒資料：

| 問題 | 影響 |
|---|---|
| 592 筆 label 為小寫 `spam` | 訓練腳本 `isin(LABELS)` 直接丟棄，UCI 翻譯簡訊幾乎全數沒進訓練 |
| ABUSIVE 有 9,223 筆來自 Jigsaw（英文維基討論頁）翻譯，佔 ABUSIVE 的 90% | 語境與客服無關、長文、大量維基編輯語境；ABUSIVE（10,268）反而多於 NORMAL（5,917） |
| PROMPT_ATTACK 有 1,553 筆是 JailBreakV 的「圖片底部顯示…」多模態題（缺圖） | 多為無害問題（如「如何保護這項權利」），卻被標成攻擊 |
| 翻譯失敗殘留英文 1,282 筆、翻譯/模板前綴（"OpenAI 的內容政策…"、"I will translate…"）184 筆、GCG 亂碼後綴 76 筆、ChatGPT/OPPO 雙回答殘骸 266 筆 | 雜訊（step06 翻譯失敗時保留原文） |
| 翻譯來源 PROMPT_ATTACK 中 613 筆沒有任何注入/越獄訊號（多為仇恨言論或一般有害提問） | 類別定義混淆 |
| NORMAL 全部是 ML_data 模板句，最長 26 字；PROMPT_ATTACK/ABUSIVE 中位數 33～44 字 | 模型學到「長訊息 ≠ NORMAL」的長度捷徑，真實長句客服提問會被誤判 |
| NORMAL 模板大量近重複（去填充詞後 1,500+ 筆重複），切分前未做近重複去重 | 測試集與訓練集高度重疊，報告中 99～100% 的分數偏樂觀 |
| SPAM 模板誤組出 31 筆合法通知（「若非本人請忽略」、「請備份資料」） | 標籤錯誤 |
| 完全缺少「困難負樣本」：含立即/投資/加入/垃圾/禮品卡/網址、轉貼可疑簡訊詢問真偽等正常提問 | 與 guardrail 規則誤判同一類問題 |

處理方式：`data_pipeline/step09_clean_dataset.py` → `data_set/train_full_dataset_v2.csv`（10,533 筆：NORMAL 4,691 / ABUSIVE 3,639 / SPAM 1,458 / PROMPT_ATTACK 745）。
每一筆被移除的資料與原因在 `data_set/removed_rows_v2.csv`；另加入 `data_pipeline/curated_seed_data.py` + `curated_seed_extra.py` 的 537 筆人工撰寫樣本（NORMAL 287 / ABUSIVE 104 / PROMPT_ATTACK 88 / SPAM 58；涵蓋電商、物流、電信、3C、金融保險、訂閱、旅遊餐飲、教育，重點是長句與含敏感字眼的困難負樣本、有情緒但無髒話的抱怨、中文客服情境的注入與越權）。
`data_set/hard_eval.csv` 為困難評估集（47 筆，不進訓練），重訓後請以它驗證誤判。六支訓練腳本的 `DATA_PATH` 已改指向 v2，舊檔保留供對照。

已知限制 / 後續：
- NORMAL 長句仍偏少（≥30 字只有 116 筆，其餘 4,500+ 筆是 ML_data 短模板句），長度捷徑只是減輕、尚未消除；重訓後請特別看 hard_eval 裡長句 NORMAL 的表現
- PROMPT_ATTACK 只剩 745 筆，且翻譯來源有同一越獄模板的多個翻譯變體 → 建議再補中文客服情境的注入樣本
- 標籤定義（2026-11-26 確認）：ABUSIVE = 使用者「有情緒」（不滿、焦急、失望、憤怒，有無髒話皆算），策略是先安撫再回答；ML_data 中「客服裝死不理人！準備接客訴」這類激動抱怨標 ABUSIVE 是正確的。人工種子已依此定義標註
- v2 尚未重新訓練；上線模型仍是舊資料訓練的版本
