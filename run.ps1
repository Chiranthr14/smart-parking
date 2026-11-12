# SmartPark PowerShell Single-Command Launcher
Set-Location -Path $PSScriptRoot

if (Test-Path ".\.venv\Scripts\python.exe") {
    & ".\.venv\Scripts\python.exe" run.py
} else {
    python run.py
}
