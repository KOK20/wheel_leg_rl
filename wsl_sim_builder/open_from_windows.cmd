@echo off
setlocal

set "URL=http://localhost:18765"
set "PROJECT=/home/gyj/parallel_wheel_leg_mujoco"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $health = Invoke-RestMethod -Uri '%URL%/api/health' -TimeoutSec 1; $ready = ($health.service -eq 'wsl_sim_builder') } catch { $ready=$false }; if (-not $ready) { Start-Process -FilePath wsl.exe -ArgumentList @('bash','-lc','cd %PROJECT% && .venv/bin/python wsl_sim_builder/server.py --host 0.0.0.0 --port 18765') -WindowStyle Minimized; Start-Sleep -Seconds 2 }; Start-Process '%URL%'"

if errorlevel 1 (
  start "WSL MuJoCo Server" wsl.exe bash -lc "cd %PROJECT% && .venv/bin/python wsl_sim_builder/server.py --host 0.0.0.0 --port 18765"
  timeout /t 2 /nobreak >nul
  start "" "%URL%"
)
