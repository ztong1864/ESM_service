@echo off
REM Double-click to start the ESM service locally (no Docker required).
REM Runs run_local.ps1, which creates/reuses .venv, fetches the model files and launches the server.

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "run_local.ps1"

echo.
echo Server stopped. Press any key to close this window.
pause >nul
