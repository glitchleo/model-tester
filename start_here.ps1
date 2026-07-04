$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "==> Preparing Deepfake Model Tester" -ForegroundColor Cyan
& powershell -ExecutionPolicy Bypass -File .\setup_submission.ps1

Write-Host ""
Write-Host "==> Starting web app" -ForegroundColor Cyan
Set-Location .\app\web
& npm run dev
