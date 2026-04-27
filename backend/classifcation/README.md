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
[NORMAL] threshold(from val) = 0.64
              precision    recall  f1-score   support

           0     0.9727    0.9727    0.9727       110
           1     0.9971    0.9971    0.9971      1028

    accuracy                         0.9947      1138
   macro avg     0.9849    0.9849    0.9849      1138
weighted avg     0.9947    0.9947    0.9947      1138

[ABUSIVE] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     0.9982    0.9991    0.9986      1105
           1     0.9688    0.9394    0.9538        33

    accuracy                         0.9974      1138
   macro avg     0.9835    0.9692    0.9762      1138
weighted avg     0.9973    0.9974    0.9973      1138

[PROMPT_ATTACK] threshold(from val) = 0.48
              precision    recall  f1-score   support

           0     1.0000    0.9991    0.9996      1113
           1     0.9615    1.0000    0.9804        25

    accuracy                         0.9991      1138
   macro avg     0.9808    0.9996    0.9900      1138
weighted avg     0.9992    0.9991    0.9991      1138

[SPAM] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     0.9972    0.9991    0.9982      1086
           1     0.9800    0.9423    0.9608        52

    accuracy                         0.9965      1138
   macro avg     0.9886    0.9707    0.9795      1138
weighted avg     0.9965    0.9965    0.9965      1138
```

---
## LR（多類）

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

saved => E:\CS_Agent\backend\backend_ml_ovr_models\normal_bin.joblib

========================================================================
Training binary model: ABUSIVE vs NOT_ABUSIVE
[ABUSIVE] best threshold = 0.82
              precision    recall  f1-score   support

           0     0.9973    0.9991    0.9982      1105
           1     0.9677    0.9091    0.9375        33

    accuracy                         0.9965      1138
   macro avg     0.9825    0.9541    0.9678      1138
weighted avg     0.9964    0.9965    0.9964      1138

saved => E:\CS_Agent\backend\backend_ml_ovr_models\abusive_bin.joblib

========================================================================
Training binary model: PROMPT_ATTACK vs NOT_PROMPT_ATTACK
[PROMPT_ATTACK] best threshold = 0.66
              precision    recall  f1-score   support

           0     0.9991    1.0000    0.9996      1113
           1     1.0000    0.9600    0.9796        25

    accuracy                         0.9991      1138
   macro avg     0.9996    0.9800    0.9896      1138
weighted avg     0.9991    0.9991    0.9991      1138

saved => E:\CS_Agent\backend\backend_ml_ovr_models\prompt_attack_bin.joblib

========================================================================
Training binary model: SPAM vs NOT_SPAM
[SPAM] best threshold = 0.52
              precision    recall  f1-score   support

           0     0.9982    0.9991    0.9986      1086
           1     0.9804    0.9615    0.9709        52

    accuracy                         0.9974      1138
   macro avg     0.9893    0.9803    0.9847      1138
weighted avg     0.9973    0.9974    0.9974      1138

saved => E:\CS_Agent\backend\backend_ml_ovr_models\spam_bin.joblib

========================================================================
Saved config => E:\CS_Agent\backend\backend_ml_ovr_models\ovr_config.json
Thresholds: {'NORMAL': 0.32, 'ABUSIVE': 0.82, 'PROMPT_ATTACK': 0.66, 'SPAM': 0.52}
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
[NORMAL] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000       110
           1     1.0000    1.0000    1.0000      1028

    accuracy                         1.0000      1138
   macro avg     1.0000    1.0000    1.0000      1138
weighted avg     1.0000    1.0000    1.0000      1138

[ABUSIVE] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     1.0000    0.9991    0.9995      1105
           1     0.9706    1.0000    0.9851        33

    accuracy                         0.9991      1138
   macro avg     0.9853    0.9995    0.9923      1138
weighted avg     0.9991    0.9991    0.9991      1138

[PROMPT_ATTACK] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     0.9991    1.0000    0.9996      1113
           1     1.0000    0.9600    0.9796        25

    accuracy                         0.9991      1138
   macro avg     0.9996    0.9800    0.9896      1138
weighted avg     0.9991    0.9991    0.9991      1138

[SPAM] threshold(from val) = 0.3
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000      1086
           1     1.0000    1.0000    1.0000        52

    accuracy                         1.0000      1138
   macro avg     1.0000    1.0000    1.0000      1138
weighted avg     1.0000    1.0000    1.0000      1138
```
