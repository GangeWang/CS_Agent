# 依序訓練六種分類方式（資料集：data_set/train_full_dataset.csv）
# 原名 batch_run.ps1，改名以符合用途；腳本清單同步新版命名
Set-Location "E:\CS_Agent\backend\classifcation"
$py = "..\.venv\Scripts\python.exe"
$log = "all_output.txt"

"" | Out-File $log

$scripts = @(
    "ML\train_multiclass_ml.py",
    "ML\train_ovr_ml.py",
    "BILSTM\train_multiclass_bilstm.py",
    "BILSTM\train_ovr_bilstm.py",
    "transformer\train_multiclass_transformer_legacy.py",
    "transformer\train_ovr_transformer.py"
)

foreach ($s in $scripts) {
    "========== Running $s ==========" | Tee-Object -FilePath $log -Append
    & $py $s 2>&1 | Tee-Object -FilePath $log -Append
    "" | Tee-Object -FilePath $log -Append
}

"Done!" | Tee-Object -FilePath $log -Append
