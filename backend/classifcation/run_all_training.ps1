# 依序訓練六種分類方式（資料集：data_set/train_full_dataset_v2.csv（step09 清理版））
# 各腳本的資料切分、評估與輸出路徑由 common.py 統一（不受執行目錄影響），報告寫到 reports/<method>.json，
# 最後以 compare_reports.py 並排比較，並用 evaluate_guardrail_pipeline.py 評估上線完整流程。
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$log = "all_output.txt"

"" | Out-File $log

$scripts = @(
    "ML\train_multiclass_ml.py",
    "ML\train_ovr_ml.py",
    "BILSTM\train_multiclass_bilstm.py",
    "BILSTM\train_ovr_bilstm.py",
    "transformer\train_multiclass_transformer_legacy.py",
    "transformer\train_ovr_transformer.py",
    "evaluate_guardrail_pipeline.py",
    "compare_reports.py"
)

foreach ($s in $scripts) {
    "========== Running $s ==========" | Tee-Object -FilePath $log -Append
    & $py $s 2>&1 | Tee-Object -FilePath $log -Append
    "" | Tee-Object -FilePath $log -Append
}

"Done!" | Tee-Object -FilePath $log -Append
