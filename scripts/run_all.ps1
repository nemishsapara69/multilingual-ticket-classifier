param(
    [switch]$SkipInstall,
    [switch]$SkipTrain,
    [switch]$RunEval,
    [switch]$ApiOnly
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host "`n=== $msg ===" -ForegroundColor Cyan
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$py = "C:/Users/nemis/AppData/Local/Programs/Python/Python313/python.exe"

Write-Step "Project root: $repoRoot"

if (-not $SkipInstall) {
    Write-Step "Installing Python dependencies"
    & $py -m pip install -r requirements.txt

    if (-not $ApiOnly) {
        Write-Step "Installing React dependencies"
        Push-Location frontend
        npm install
        Pop-Location
    }
}

if (-not $SkipTrain) {
    Write-Step "Preprocessing dataset"
    & $py -m src.data_preprocessing --input data/raw/tickets.csv --output_dir data/processed

    Write-Step "Training model"
    & $py -m src.train --data_dir data/processed --model_dir models/best --params params.yaml --metrics_path models/metrics.json
}

Write-Step "Starting API server in background"
$apiCmd = "$py -m uvicorn src.api:app --host 0.0.0.0 --port 8000"
$apiProc = Start-Process -FilePath powershell -ArgumentList "-NoProfile", "-Command", $apiCmd -PassThru
Start-Sleep -Seconds 3

if ($RunEval) {
    Write-Step "Running evaluations against API"
    & $py -m src.evaluate --mode api --input data/eval/eval_queries.csv --output reports/eval_report.json
    & $py -m src.evaluate --mode api --input data/eval/eval_queries_realworld.csv --output reports/eval_report_realworld.json
}

if ($ApiOnly) {
    Write-Step "API is running"
    Write-Host "API URL: http://localhost:8000" -ForegroundColor Green
    Write-Host "To stop API process: Stop-Process -Id $($apiProc.Id)" -ForegroundColor Yellow
    exit 0
}

Write-Step "Starting React web app"
Push-Location frontend
npm run dev
Pop-Location
