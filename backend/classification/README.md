# Guardrail 模型結果整理

> 最新結果：2026-11-26 第四批樣本重訓（v2 = 11,128 筆，負責人 GPU 機器跑 `run_all_training.ps1`）。數字取自 `reports/*.json`，重跑後可用 `python compare_reports.py` 重新產生比較表。
> test = 共用切分的 20%（2,226 筆，seed 42，近重複已去除）；hard_eval = `data_set/hard_eval.csv` 199 句困難樣本（不進訓練）。
> OvR 的 test / hard_eval 都是以上線決策規則（過門檻取最高、都沒過 → UNCERTAIN）組合成 4 類後的結果，可與 4選1 直接比較。
> 舊版（v1 資料、1,138 筆 test、未去近重複）的 99～100% 分數偏樂觀且不能與這裡比較，已移除；需要時用 `git show 117b544:backend/classification/README.md` 查看。

## 總覽

| 方法 | test macro-F1 | test 準確率 | test UNCERTAIN | hard_eval macro-F1 | hard_eval 準確率 | hard_eval 判錯 | hard_eval UNCERTAIN |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR 4選1 | 0.953 | 0.969 | — | 0.821 | 0.829 | 34 / 199 | — |
| TF-IDF + LR OvR | 0.942 | 0.960 | 0.7% | 0.815 | 0.804 | 39 / 199 | 7.0% |
| BiLSTM 4選1 | 0.942 | 0.964 | — | 0.835 | 0.844 | 31 / 199 | — |
| BiLSTM OvR | 0.938 | 0.953 | 2.9% | 0.741 | 0.729 | 54 / 199 | 15.6% |
| Transformer 4選1 | 0.987 | 0.992 | — | 0.984 | 0.985 | 3 / 199 | — |
| Transformer OvR（上線模型） | 0.991 | 0.989 | 0.4% | 0.949 | 0.935 | 13 / 199 | 2.5% |
| **上線流程**（規則 → Transformer OvR → 語意比對） | 0.992 | 0.992 | 0.1% | 0.956 | 0.950 | 10 / 199 | 1.0% |

重點：

- **上線流程 hard_eval macro-F1 0.918 → 0.956**（判錯 17 → 10 句，含 2 句 UNCERTAIN），test 0.994 → 0.992。
- 第四批樣本是針對 hard_eval 判錯的情境寫的，hard_eval 現在比較像開發集，分數偏樂觀；要量真實表現需要另一份沒拿來補樣本的盲測集。
- test 分數比第三次重訓略降不是退步：共用切分把部分第四批樣本分進 test，這些句子比模板句難。
- Transformer 4選1 在 hard_eval 上（0.984）高於 OvR（0.949），上線依負責人決定維持 OvR。
- 上線流程仍判錯的 10 句：英文無髒話的情緒抱怨 4 句被判 NORMAL；NORMAL 被誤判 4 句（問 AI 能不能記得對話、要求轉真人 → PROMPT_ATTACK；轉貼 LINE 假業務、家人收到中獎簡訊來確認 → PROMPT_ATTACK / SPAM）；英文冒充管理員要折扣碼、離題閒聊天氣各 1 句落入 UNCERTAIN。完整清單在 `reports/pipeline.json` 的 `hard_eval.errors`。

### OvR 門檻（在 val 上選出）

| 方法 | NORMAL | ABUSIVE | PROMPT_ATTACK | SPAM |
|---|---|---|---|---|
| TF-IDF + LR OvR | 0.25 | 0.59 | 0.74 | 0.47 |
| BiLSTM OvR | 0.48 | 0.41 | 0.93 | 0.87 |
| Transformer OvR（上線模型） | 0.23 | 0.61 | 0.41 | 0.95 |

上線流程使用 Transformer OvR 的門檻。

---

## TF-IDF + LR 4選1

### test（2,226 筆）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9938 | 0.9748 | 0.9842 | 992 |
| ABUSIVE | 0.9551 | 0.9841 | 0.9694 | 756 |
| PROMPT_ATTACK | 0.9006 | 0.8851 | 0.8928 | 174 |
| SPAM | 0.9670 | 0.9638 | 0.9654 | 304 |
| **macro avg** | 0.9541 | 0.9519 | **0.9529** | 2226 |

準確率 0.9695

### hard_eval（199 句）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 1.0000 | 0.7595 | 0.8633 | 79 |
| ABUSIVE | 0.7541 | 0.9200 | 0.8288 | 50 |
| PROMPT_ATTACK | 0.7727 | 0.8500 | 0.8095 | 40 |
| SPAM | 0.7353 | 0.8333 | 0.7812 | 30 |
| **macro avg** | 0.8155 | 0.8407 | **0.8207** | 199 |

準確率 0.8291

| 真實 \ 預測 | NORMAL | ABUSIVE | PROMPT_ATTACK | SPAM | UNCERTAIN |
|---|---|---|---|---|---|
| NORMAL | 60 | 6 | 6 | 7 | 0 |
| ABUSIVE | 0 | 46 | 3 | 1 | 0 |
| PROMPT_ATTACK | 0 | 5 | 34 | 1 | 0 |
| SPAM | 0 | 4 | 1 | 25 | 0 |

---

## TF-IDF + LR OvR

### test（2,226 筆）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9721 | 0.9849 | 0.9785 | 992 |
| ABUSIVE | 0.9609 | 0.9749 | 0.9678 | 756 |
| PROMPT_ATTACK | 0.9854 | 0.7759 | 0.8682 | 174 |
| SPAM | 0.9601 | 0.9507 | 0.9554 | 304 |
| **macro avg** | 0.9696 | 0.9216 | **0.9425** | 2226 |

準確率 0.9605，UNCERTAIN 0.7%

### hard_eval（199 句）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9429 | 0.8354 | 0.8859 | 79 |
| ABUSIVE | 0.7895 | 0.9000 | 0.8411 | 50 |
| PROMPT_ATTACK | 0.9615 | 0.6250 | 0.7576 | 40 |
| SPAM | 0.7500 | 0.8000 | 0.7742 | 30 |
| **macro avg** | 0.8610 | 0.7901 | **0.8147** | 199 |

準確率 0.8040，UNCERTAIN 7.0%

| 真實 \ 預測 | NORMAL | ABUSIVE | PROMPT_ATTACK | SPAM | UNCERTAIN |
|---|---|---|---|---|---|
| NORMAL | 66 | 3 | 1 | 4 | 5 |
| ABUSIVE | 0 | 45 | 0 | 2 | 3 |
| PROMPT_ATTACK | 3 | 6 | 25 | 2 | 4 |
| SPAM | 1 | 3 | 0 | 24 | 2 |

### 各二元分類器（test）

```text
[NORMAL] threshold = 0.25
              precision    recall  f1-score   support

           0     0.9941    0.9595    0.9765      1234
           1     0.9517    0.9929    0.9719       992

    accuracy                         0.9744      2226
   macro avg     0.9729    0.9762    0.9742      2226
weighted avg     0.9752    0.9744    0.9744      2226

[ABUSIVE] threshold = 0.59
              precision    recall  f1-score   support

           0     0.9870    0.9796    0.9833      1470
           1     0.9609    0.9749    0.9678       756

    accuracy                         0.9780      2226
   macro avg     0.9739    0.9772    0.9755      2226
weighted avg     0.9781    0.9780    0.9780      2226

[PROMPT_ATTACK] threshold = 0.74
              precision    recall  f1-score   support

           0     0.9818    0.9990    0.9903      2052
           1     0.9855    0.7816    0.8718       174

    accuracy                         0.9820      2226
   macro avg     0.9837    0.8903    0.9311      2226
weighted avg     0.9821    0.9820    0.9811      2226

[SPAM] threshold = 0.47
              precision    recall  f1-score   support

           0     0.9927    0.9912    0.9919      1922
           1     0.9446    0.9539    0.9493       304

    accuracy                         0.9861      2226
   macro avg     0.9687    0.9726    0.9706      2226
weighted avg     0.9861    0.9861    0.9861      2226
```

---

## BiLSTM 4選1

### test（2,226 筆）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9868 | 0.9798 | 0.9833 | 992 |
| ABUSIVE | 0.9797 | 0.9590 | 0.9693 | 756 |
| PROMPT_ATTACK | 0.8556 | 0.8851 | 0.8701 | 174 |
| SPAM | 0.9190 | 0.9704 | 0.9440 | 304 |
| **macro avg** | 0.9353 | 0.9486 | **0.9417** | 2226 |

準確率 0.9641

### hard_eval（199 句）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9189 | 0.8608 | 0.8889 | 79 |
| ABUSIVE | 0.8409 | 0.7400 | 0.7872 | 50 |
| PROMPT_ATTACK | 0.7872 | 0.9250 | 0.8506 | 40 |
| SPAM | 0.7647 | 0.8667 | 0.8125 | 30 |
| **macro avg** | 0.8279 | 0.8481 | **0.8348** | 199 |

準確率 0.8442

| 真實 \ 預測 | NORMAL | ABUSIVE | PROMPT_ATTACK | SPAM | UNCERTAIN |
|---|---|---|---|---|---|
| NORMAL | 68 | 3 | 3 | 5 | 0 |
| ABUSIVE | 4 | 37 | 6 | 3 | 0 |
| PROMPT_ATTACK | 2 | 1 | 37 | 0 | 0 |
| SPAM | 0 | 3 | 1 | 26 | 0 |

---

## BiLSTM OvR

### test（2,226 筆）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9929 | 0.9819 | 0.9873 | 992 |
| ABUSIVE | 0.9723 | 0.9749 | 0.9736 | 756 |
| PROMPT_ATTACK | 0.9632 | 0.7529 | 0.8452 | 174 |
| SPAM | 0.9721 | 0.9178 | 0.9442 | 304 |
| **macro avg** | 0.9751 | 0.9068 | **0.9376** | 2226 |

準確率 0.9528，UNCERTAIN 2.9%

### hard_eval（199 句）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9718 | 0.8734 | 0.9200 | 79 |
| ABUSIVE | 0.7321 | 0.8200 | 0.7736 | 50 |
| PROMPT_ATTACK | 0.8235 | 0.3500 | 0.4912 | 40 |
| SPAM | 0.8750 | 0.7000 | 0.7778 | 30 |
| **macro avg** | 0.8506 | 0.6859 | **0.7406** | 199 |

準確率 0.7286，UNCERTAIN 15.6%

| 真實 \ 預測 | NORMAL | ABUSIVE | PROMPT_ATTACK | SPAM | UNCERTAIN |
|---|---|---|---|---|---|
| NORMAL | 69 | 4 | 0 | 2 | 4 |
| ABUSIVE | 2 | 41 | 2 | 0 | 5 |
| PROMPT_ATTACK | 0 | 7 | 14 | 1 | 18 |
| SPAM | 0 | 4 | 1 | 21 | 4 |

### 各二元分類器（test）

```text
[NORMAL] threshold = 0.48
              precision    recall  f1-score   support

           0     0.9878    0.9878    0.9878      1234
           1     0.9849    0.9849    0.9849       992

    accuracy                         0.9865      2226
   macro avg     0.9864    0.9864    0.9864      2226
weighted avg     0.9865    0.9865    0.9865      2226

[ABUSIVE] threshold = 0.41
              precision    recall  f1-score   support

           0     0.9890    0.9810    0.9850      1470
           1     0.9635    0.9788    0.9711       756

    accuracy                         0.9802      2226
   macro avg     0.9763    0.9799    0.9781      2226
weighted avg     0.9804    0.9802    0.9803      2226

[PROMPT_ATTACK] threshold = 0.93
              precision    recall  f1-score   support

           0     0.9794    0.9966    0.9879      2052
           1     0.9493    0.7529    0.8397       174

    accuracy                         0.9775      2226
   macro avg     0.9643    0.8747    0.9138      2226
weighted avg     0.9771    0.9775    0.9763      2226

[SPAM] threshold = 0.87
              precision    recall  f1-score   support

           0     0.9871    0.9943    0.9907      1922
           1     0.9621    0.9178    0.9394       304

    accuracy                         0.9838      2226
   macro avg     0.9746    0.9560    0.9650      2226
weighted avg     0.9837    0.9838    0.9837      2226
```

---

## Transformer 4選1

### test（2,226 筆）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9940 | 0.9970 | 0.9955 | 992 |
| ABUSIVE | 0.9973 | 0.9881 | 0.9927 | 756 |
| PROMPT_ATTACK | 0.9659 | 0.9770 | 0.9714 | 174 |
| SPAM | 0.9869 | 0.9934 | 0.9902 | 304 |
| **macro avg** | 0.9860 | 0.9889 | **0.9874** | 2226 |

準確率 0.9919

### hard_eval（199 句）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9873 | 0.9873 | 0.9873 | 79 |
| ABUSIVE | 0.9804 | 1.0000 | 0.9901 | 50 |
| PROMPT_ATTACK | 0.9750 | 0.9750 | 0.9750 | 40 |
| SPAM | 1.0000 | 0.9667 | 0.9831 | 30 |
| **macro avg** | 0.9857 | 0.9823 | **0.9839** | 199 |

準確率 0.9849

| 真實 \ 預測 | NORMAL | ABUSIVE | PROMPT_ATTACK | SPAM | UNCERTAIN |
|---|---|---|---|---|---|
| NORMAL | 78 | 1 | 0 | 0 | 0 |
| ABUSIVE | 0 | 50 | 0 | 0 | 0 |
| PROMPT_ATTACK | 1 | 0 | 39 | 0 | 0 |
| SPAM | 0 | 0 | 1 | 29 | 0 |

---

## Transformer OvR（上線模型）

### test（2,226 筆）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9920 | 0.9960 | 0.9940 | 992 |
| ABUSIVE | 0.9973 | 0.9828 | 0.9900 | 756 |
| PROMPT_ATTACK | 0.9831 | 1.0000 | 0.9915 | 174 |
| SPAM | 0.9966 | 0.9770 | 0.9867 | 304 |
| **macro avg** | 0.9922 | 0.9889 | **0.9905** | 2226 |

準確率 0.9892，UNCERTAIN 0.4%

### hard_eval（199 句）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9487 | 0.9367 | 0.9427 | 79 |
| ABUSIVE | 1.0000 | 0.8800 | 0.9362 | 50 |
| PROMPT_ATTACK | 0.9286 | 0.9750 | 0.9512 | 40 |
| SPAM | 0.9667 | 0.9667 | 0.9667 | 30 |
| **macro avg** | 0.9610 | 0.9396 | **0.9492** | 199 |

準確率 0.9347，UNCERTAIN 2.5%

| 真實 \ 預測 | NORMAL | ABUSIVE | PROMPT_ATTACK | SPAM | UNCERTAIN |
|---|---|---|---|---|---|
| NORMAL | 74 | 0 | 3 | 1 | 1 |
| ABUSIVE | 4 | 44 | 0 | 0 | 2 |
| PROMPT_ATTACK | 0 | 0 | 39 | 0 | 1 |
| SPAM | 0 | 0 | 0 | 29 | 1 |

### 各二元分類器（test）

```text
[NORMAL] threshold = 0.23
              precision    recall  f1-score   support

           0     0.9975    0.9878    0.9927      1234
           1     0.9851    0.9970    0.9910       992

    accuracy                         0.9919      2226
   macro avg     0.9913    0.9924    0.9918      2226
weighted avg     0.9920    0.9919    0.9919      2226

[ABUSIVE] threshold = 0.61
              precision    recall  f1-score   support

           0     0.9919    0.9973    0.9946      1470
           1     0.9947    0.9841    0.9894       756

    accuracy                         0.9928      2226
   macro avg     0.9933    0.9907    0.9920      2226
weighted avg     0.9928    0.9928    0.9928      2226

[PROMPT_ATTACK] threshold = 0.41
              precision    recall  f1-score   support

           0     1.0000    0.9946    0.9973      2052
           1     0.9405    1.0000    0.9694       174

    accuracy                         0.9951      2226
   macro avg     0.9703    0.9973    0.9833      2226
weighted avg     0.9954    0.9951    0.9951      2226

[SPAM] threshold = 0.95
              precision    recall  f1-score   support

           0     0.9964    0.9974    0.9969      1922
           1     0.9834    0.9770    0.9802       304

    accuracy                         0.9946      2226
   macro avg     0.9899    0.9872    0.9885      2226
weighted avg     0.9946    0.9946    0.9946      2226
```

---

## 上線流程（規則 → Transformer OvR → 語意比對）

### test（2,226 筆）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9920 | 0.9970 | 0.9945 | 992 |
| ABUSIVE | 0.9973 | 0.9868 | 0.9920 | 756 |
| PROMPT_ATTACK | 0.9831 | 1.0000 | 0.9915 | 174 |
| SPAM | 0.9967 | 0.9868 | 0.9917 | 304 |
| **macro avg** | 0.9923 | 0.9926 | **0.9924** | 2226 |

準確率 0.9924，UNCERTAIN 0.1%

### hard_eval（199 句）

| 類別 | precision | recall | F1 | 筆數 |
|---|---|---|---|---|
| NORMAL | 0.9494 | 0.9494 | 0.9494 | 79 |
| ABUSIVE | 1.0000 | 0.9200 | 0.9583 | 50 |
| PROMPT_ATTACK | 0.9286 | 0.9750 | 0.9512 | 40 |
| SPAM | 0.9667 | 0.9667 | 0.9667 | 30 |
| **macro avg** | 0.9612 | 0.9528 | **0.9564** | 199 |

準確率 0.9497，UNCERTAIN 1.0%

| 真實 \ 預測 | NORMAL | ABUSIVE | PROMPT_ATTACK | SPAM | UNCERTAIN |
|---|---|---|---|---|---|
| NORMAL | 75 | 0 | 3 | 1 | 0 |
| ABUSIVE | 4 | 46 | 0 | 0 | 0 |
| PROMPT_ATTACK | 0 | 0 | 39 | 0 | 1 |
| SPAM | 0 | 0 | 0 | 29 | 1 |

test 各階段判定筆數：ML 2171、RULE 46、SEMANTIC 6、UNCERTAIN 3

hard_eval 各階段判定筆數：ML 184、RULE 10、SEMANTIC 3、UNCERTAIN 2

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

- **入庫的只有原始來源與翻譯結果**（2026-11-26 清理）：外部原始資料（HF parquet、`JailBreakV_28K.csv`、`sms-spam-collection.csv`）、
  翻譯產物（`*_translated_zh.csv`，ollama 翻譯無法便宜重現）、`data_set/` 底下的檔案。
  抽取與合併的中間檔中，`extracted_abusive.csv`（step01）、`extracted_spam.csv`（step03）、`merged_full_dataset.csv`（step07，只差換行符號）
  重跑結果與歷史檔一致；**prompt attack 這條無法用目前的腳本重現**：照現在的 step02 會抽出 28,000 筆（含重複），
  歷史的 `extracted_prompt_attack.csv` 卻是 5,168 筆不重複、其中 863 筆文字與現在的 `JailBreakV_28K.csv` 略有出入，
  `deduped_prompt_attack.csv` 也有 206 筆不在那個歷史抽取檔裡，當時的實際做法沒有留下紀錄。
  下游只用到 `prompt_attack_translated_zh.csv`（已入庫），所以不影響訓練；要重做翻譯時請從 git 歷史取回原檔，不要重跑 step02/04。
  `deduped_full_dataset.csv` 與 `data_set/train_full_dataset.csv` 內容完全相同（md5 一致）。這些檔案和從沒被任何腳本用到的
  `spml_prompt_injection.csv` 都已從目前版本移除，需要時可從 git 歷史取回：
  `git checkout ba4c63a -- backend/classifcation/<檔名>`（那時目錄名還是拼錯的 `classifcation`，取回後搬到 `backend/classification/`）
- 各外部來源一路保留了多少（2026-11-26 統計；v2 共 11,128 筆，其中外部翻譯來源 3,256 筆，其餘為 `ML_data.csv` 6,744 筆、人工樣本 1,128 筆）：

  | 來源 | 原始檔 | 抽取 | 翻譯 | step08 去重後 | v2 最終 |
  |---|---|---|---|---|---|
  | HF parquet（維基討論頁留言） | 159,100 筆（有毒 9,602 / 無毒 149,498） | 有毒 9,433 | 9,433（692 筆翻譯失敗仍是英文） | 9,223 | ABUSIVE 2,500（下採樣上限） |
  | `JailBreakV_28K.csv` | 28,000 筆（不重複 8,641） | 5,168 → step04 4,172 | 4,172（740 筆仍是英文） | 3,351 | PROMPT_ATTACK 196 |
  | `sms-spam-collection.csv` | 5,572 筆（spam 747 / ham 4,825） | spam 747 | 747（32 筆仍是英文） | 592 | SPAM 560 |

  入庫的原始檔沒有被截短：JailBreakV 28,000 筆、SMS 5,572 筆都是公開版的完整筆數；parquet 是某個 HF 資料集的 train 分片，上游筆數無法從這裡連線核對。無毒留言與 ham 簡訊從來沒用上。step09 移除的每一筆和原因都在 `data_set/removed_rows_v2.csv`。
- 目錄原名 `classifcation`（拼錯），2026-11-26 改成 `classification`。
- 歷史檔名對照：`full_data_set_fillted.csv`（filtered 錯字）= 現 `train_full_dataset.csv`；`FILLTTER/` = 現 `data_pipeline/`
- 合併與翻譯步驟原為手動/遺失腳本，step06/07 為事後補寫使管線可完整重現
- 已知資料特性：去重後仍有 592 筆 label 為小寫 `spam`（其餘為大寫 `SPAM`），訓練/評估時注意
- 完整實驗結果見本檔開頭的總覽與各方法報告

---

## 六種分類方式矩陣（3 演算法 × 2 架構）

| | 4選1（單模型多分類） | OVR（4 個二元分類器，每 label 一個模型） |
|---|---|---|
| **ML**（TF-IDF + LR） | `ML/train_multiclass_ml.py` → `backend_ml_guardrail_model.joblib` | `ML/train_ovr_ml.py` → `ML/backend_ml_ovr_models/`（4 個 _bin.joblib） |
| **BiLSTM**（字元級，packed） | `BILSTM/train_multiclass_bilstm.py` → `backend_ml_guardrail_bilstm_pytorch.pt` + vocab + label_encoder | `BILSTM/train_ovr_bilstm.py` → `BILSTM/backend_ml_ovr_bilstm_models/`（4 .pt + vocab） |
| **Transformer**（hfl/chinese-roberta-wwm-ext） | `transformer/train_multiclass_transformer.py` → `transformer/backend_ml_guardrail_transformer/` | `transformer/train_ovr_transformer.py` → `transformer/backend_ml_ovr_transformer_models_clean/` 🏆 **上線模型** |

### 訓練與評估（2026-11-26 起）

六支腳本共用 `common.py`：同一份資料（`data_set/train_full_dataset_v2.csv`）、同一個切分（train 72% / val 8% / test 20%，seed 42，近重複已去除）、
同一套評估，輸出路徑固定（不受執行目錄影響）。每支腳本都會：

1. 在 **val** 上選門檻（OvR；F1 平台取中位數，平台過寬會印 `[WARN]`）
2. 在 **test** 上報告；OvR 以上線決策規則（過門檻取最高、都沒過 → UNCERTAIN）組合成 4 類結果，與 4選1 可直接比較
3. 在 **`data_set/hard_eval.csv`**（困難樣本，不進訓練）上報告並列出錯誤樣本
4. 寫入 `reports/<method>.json`

```powershell
.\run_all_training.ps1                       # 六種方法 + 重建語意庫 + 上線流程評估 + 比較表（-SkipTest：評估只跑 hard_eval）
python compare_reports.py                    # 只看比較表
python evaluate_guardrail_pipeline.py        # 評估上線完整流程（RULE → ML → SEMANTIC）
$env:GUARDRAIL_BASE_MODEL="ckiplab/bert-base-chinese"; python transformer\train_ovr_transformer.py  # 換預訓練模型 A/B，輸出到另一個目錄，不覆蓋上線模型
```

`run_all_training.ps1` 會在端到端評估之前重建語意庫（`build_semantic_models.py`，已改用 v2 資料）。單獨重訓上線模型時，記得也要重建。

Windows PowerShell 5.1 下，python 的警告會顯示成紅字 `NativeCommandError`，不代表失敗；`run_all_training.ps1` 會以結束代碼判斷，最後列出每一步成功或失敗。

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

處理方式：`data_pipeline/step09_clean_dataset.py` → `data_set/train_full_dataset_v2.csv`（11,128 筆：NORMAL 4,961 / ABUSIVE 3,779 / SPAM 1,519 / PROMPT_ATTACK 869）。
每一筆被移除的資料與原因在 `data_set/removed_rows_v2.csv`；另加入 `data_pipeline/curated_seed_data.py` + `curated_seed_extra.py` 的 537 筆人工撰寫樣本（NORMAL 287 / ABUSIVE 104 / PROMPT_ATTACK 88 / SPAM 58；涵蓋電商、物流、電信、3C、金融保險、訂閱、旅遊餐飲、教育，重點是長句與含敏感字眼的困難負樣本、有情緒但無髒話的抱怨、中文客服情境的注入與越權）。第三批 `curated_seed_long.py` 再補 185 句 30 字以上的 NORMAL 與 37 句英文樣本（四類都有）。第四批 `curated_seed_targeted.py` 依 199 句 hard_eval 上線流程判錯的情境補 373 句：沒有髒話的情緒（反諷、反問、長篇抱怨）、中文冒充權限與套取他人資料、英文、離題閒聊，以及避免矯枉過正的 NORMAL 對照組。共 1,132 筆人工樣本。
`data_set/hard_eval.csv` 為困難評估集（不進訓練），重訓後請以它驗證誤判。原本 47 筆，2026-11-26 以 `data_pipeline/curated_hard_eval.py` 擴充到 199 筆（NORMAL 79 / ABUSIVE 50 / PROMPT_ATTACK 40 / SPAM 30），擴充前後的分數不能直接比較。第四批訓練樣本是針對這份的弱點寫的，之後它比較像開發集、分數偏樂觀；要量真實表現需要另一份沒拿來補樣本的盲測集。六支訓練腳本的 `DATA_PATH` 已改指向 v2，舊檔保留供對照。

已知限制 / 後續：
- NORMAL 長句仍偏少（≥30 字 350 筆，原本 116 筆；其餘 4,500+ 筆是 ML_data 短模板句），長度捷徑只是減輕、尚未消除；重訓後請特別看 hard_eval 裡長句 NORMAL 的表現
- 離題閒聊、笑話依負責人決定維持 SPAM（拒答）
- PROMPT_ATTACK 只剩 745 筆，且翻譯來源有同一越獄模板的多個翻譯變體 → 建議再補中文客服情境的注入樣本
- 標籤定義（2026-11-26 確認）：ABUSIVE = 使用者「有情緒」（不滿、焦急、失望、憤怒，有無髒話皆算），策略是先安撫再回答；ML_data 中「客服裝死不理人！準備接客訴」這類激動抱怨標 ABUSIVE 是正確的。人工種子已依此定義標註
- v2 已於 2026-11-26 重訓四次，上線模型是第四批樣本的版本，成績見本檔開頭
