# Аналог Makefile для Windows (PowerShell), где make не установлен.
# Примеры:
#   .\tasks.ps1 predict -In tickets.jsonl -Out pred.jsonl
#   .\tasks.ps1 train | test | evaluate | run | dashboard
#   .\tasks.ps1 demo          # проверка сценария + API (:8000) + админка (:8501) в демо-режиме
#   .\tasks.ps1 demo-check    # только проверка демо-сценария
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("install", "analyze", "train", "predict", "test", "evaluate", "run", "dashboard", "demo", "demo-check", "safety-audit")]
    [string]$Task,
    [string]$In = "data/raw/handout/tickets_valid.jsonl",
    [string]$Out = "pred.jsonl",
    [int]$Port = 8000
)
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
$env:OPENBLAS_NUM_THREADS = "1"   # при 8 ГБ ОЗУ два процесса с многопоточным OpenBLAS не помещаются в память
switch ($Task) {
    "install"      { & $py -m pip install -r requirements.txt }
    "analyze"      { & $py scripts/analyze_data.py }
    "train"        { & $py scripts/prepare_dataset.py; if ($?) { & $py scripts/train_classifier.py --approaches A,B,C } }
    "predict"      { & $py scripts/predict.py --in $In --out $Out }
    "test"         { & $py -m pytest -q tests }
    "evaluate"     { & $py scripts/evaluate.py }
    "safety-audit" { & $py scripts/safety_audit.py }
    "run"          { & $py -m uvicorn app.main:app --host 0.0.0.0 --port $Port }
    "dashboard"    { & $py -m streamlit run dashboard/app.py }
    "demo-check"   { & $py scripts/demo_check.py }
    "demo" {
        & $py scripts/demo_check.py
        if ($LASTEXITCODE -ne 0) { Write-Error "Демо-сценарий не прошёл проверку — показ не запускается"; exit 1 }
        $env:DEMO = "1"
        $env:FEEDBACK_DB = "data/processed/feedback_demo.sqlite"   # действия на показе не смешиваются с рабочей базой
        $api = Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port" -PassThru -NoNewWindow
        Write-Host "API: http://127.0.0.1:$Port/docs   Админка: http://localhost:8501"
        try { & $py -m streamlit run dashboard/app.py --server.headless true }
        finally { if (-not $api.HasExited) { Stop-Process -Id $api.Id -Force } }
    }
}
exit $LASTEXITCODE
