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
$SitePackages = Join-Path $RepoRoot ".venv\Lib\site-packages"

# The environment may exist and still be incomplete: built for the tests only (they need no torch), or
# left half-installed by an interrupted download. Checking the packages that matter, rather than only the
# folder, means a service that would start degraded gets its dependencies instead.
$NeedsInstall = -not (Test-Path $VenvPython)
if (-not $NeedsInstall) {
    foreach ($package in "torch", "esm", "fastapi", "uvicorn") {
        if (-not (Test-Path (Join-Path $SitePackages $package))) { $NeedsInstall = $true }
    }
}

if ($NeedsInstall) {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating virtual environment (.venv) ..."
        python -m venv (Join-Path $RepoRoot ".venv")
    }
    # torch comes from PyTorch's CPU index: the build on PyPI drags in several GB of CUDA libraries that a
    # CPU server never uses.
    Write-Host "Installing the service's dependencies (this downloads the CPU build of torch) ..."
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    & $VenvPython -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r (Join-Path $RepoRoot "requirements.lock.txt")
}

# The model files, once. Downloading is resumable, so running this again after an interrupted download
# continues where it stopped.
& $VenvPython (Join-Path $RepoRoot "scripts\fetch_models.py")

& $VenvPython (Join-Path $RepoRoot "scripts\run_esm_api.py") --host $BindHost --port $Port
