$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PIP_NO_INDEX = $null
foreach ($ProxyVar in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")) {
    if ((Get-Item "Env:$ProxyVar" -ErrorAction SilentlyContinue).Value -eq "http://127.0.0.1:9") {
        Set-Item "Env:$ProxyVar" ""
    }
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Command
}

$PythonLauncher = "py"
$UsePyLauncher = $true

try {
    & $PythonLauncher -3.11 --version | Out-Host
} catch {
    $UsePyLauncher = $false
}

Invoke-Step "Creating Python virtual environment" {
    if ($UsePyLauncher) {
        & py -3.11 -m venv .venv
    } else {
        & python -m venv .venv
    }
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

Invoke-Step "Upgrading pip" {
    & $VenvPython -m pip install --upgrade pip
}

Invoke-Step "Installing PyTorch CUDA wheels" {
    & $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
}

Invoke-Step "Installing Python project dependencies" {
    & $VenvPython -m pip install --index-url https://pypi.org/simple -r .\requirements.txt
}

Invoke-Step "Installing React frontend dependencies" {
    Push-Location .\app\web
    try {
        if (Test-Path .\package-lock.json) {
            & npm ci
        } else {
            & npm install
        }
    } finally {
        Pop-Location
    }
}

Invoke-Step "Checking model setup" {
    & $VenvPython .\check_setup.py
}

Write-Host ""
Write-Host "Setup finished. To run the web app:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  cd .\app\web"
Write-Host "  npm run dev"
