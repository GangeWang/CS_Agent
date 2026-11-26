# 一鍵重訓與評估（資料集：data_set/train_full_dataset_v2.csv，step09 清理版）
#
# 流程：六種分類方式訓練 → 重建語意庫 → 端到端評估上線流程 → 六種方法比較表
# 各腳本的資料切分、評估與輸出路徑由 common.py 統一（不受執行目錄影響），報告寫到 reports/<method>.json。
#
# 用法（Windows PowerShell 5.1 或 PowerShell 7）：
#   .\run_all_training.ps1               # 全部
#   .\run_all_training.ps1 -SkipTest     # 端到端評估只跑 hard_eval（較快）
#
# 2026-11-26 修正：
# - stderr 轉成一般文字：PowerShell 5.1 會把 python 的 stderr（警告、進度條）包成紅字 NativeCommandError，
#   看起來像出錯但其實沒有
# - PYTHONUNBUFFERED=1：輸出經過 Tee-Object 時 python 會先暫存，畫面看起來像卡住
# - PYTHONUTF8=1 + 主控台 UTF-8：v2 有 63 句含 cp950 編不了的字，印出時可能 UnicodeEncodeError
# - GUARDRAIL_DEVICE 預設 cuda：上線 guardrail 預設用 CPU，端到端評估 2,000 多句 × 4 個模型會跑很久
# - 每支腳本檢查結束代碼，最後列出成功 / 失敗
# - 端到端評估之前先重建語意庫（原本用的是舊語意庫）
param(
    [switch]$SkipTest
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "找不到 $py ，請先在 backend\ 建立 .venv 並安裝 requirements.txt 與 requirements-dev.txt" -ForegroundColor Red
    exit 1
}

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }  # 讓 PowerShell 以 UTF-8 解讀 python 輸出
if (-not $env:GUARDRAIL_DEVICE) { $env:GUARDRAIL_DEVICE = "cuda" }  # 沒有 GPU 時 guardrail 會自動退回 CPU

$log = Join-Path $PSScriptRoot "all_output.txt"
Remove-Item $log -ErrorAction SilentlyContinue

# 寫到畫面與 all_output.txt。Out-Host 讓這些文字不會混進函式的回傳值
function Write-Log {
    process { "$_" | Tee-Object -FilePath $log -Append | Out-Host }
}

function Invoke-Step([string]$name, [string[]]$pyArgs) {
    "========== Running $name ==========" | Write-Log
    $start = Get-Date
    # "$_" 把 stderr 的 ErrorRecord 轉回一般文字，避免紅字 NativeCommandError
    & $py @pyArgs 2>&1 | Write-Log
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    $status = if ($code -eq 0) { "OK" } else { "FAILED (exit $code)" }
    "---------- $name : $status, $mins min ----------`n" | Write-Log
    return [pscustomobject]@{ Step = $name; Status = $status; Minutes = $mins }
}

& $py -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())" 2>&1 | Write-Log

$results = @()
$training = @(
    "ML\train_multiclass_ml.py",
    "ML\train_ovr_ml.py",
    "BILSTM\train_multiclass_bilstm.py",
    "BILSTM\train_ovr_bilstm.py",
    "transformer\train_multiclass_transformer_legacy.py",
    "transformer\train_ovr_transformer.py"
)
foreach ($s in $training) { $results += Invoke-Step $s @($s) }

$results += Invoke-Step "build_semantic_models.py" @("build_semantic_models.py")

$evalArgs = @("evaluate_guardrail_pipeline.py")
if ($SkipTest) { $evalArgs += "--skip-test" }
$results += Invoke-Step "evaluate_guardrail_pipeline.py" $evalArgs

$results += Invoke-Step "compare_reports.py" @("compare_reports.py")

"========== Summary ==========" | Write-Log
$results | Format-Table -AutoSize | Out-String | Write-Log
$failed = @($results | Where-Object { $_.Status -ne "OK" })
if ($failed.Count -gt 0) {
    Write-Host "有 $($failed.Count) 個步驟失敗，請在 all_output.txt 搜尋 Traceback" -ForegroundColor Red
    exit 1
}
"Done!" | Write-Log
