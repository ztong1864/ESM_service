# Run the ESM service locally without Docker.
# Mirrors the Dockerfile's CMD, using the project's own .venv.

param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8789
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment (.venv) ..."
    python -m venv (Join-Path $RepoRoot ".venv")
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    & $VenvPython -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r (Join-Path $RepoRoot "requirements.lock.txt")
}

# The model files, once. Downloading is resumable, so running this again after an interrupted download
# continues where it stopped.
& $VenvPython (Join-Path $RepoRoot "scripts\fetch_models.py")

& $VenvPython (Join-Path $RepoRoot "scripts\run_esm_api.py") --host $BindHost --port $Port
