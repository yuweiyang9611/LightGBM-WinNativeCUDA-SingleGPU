@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_portable_test.ps1" -Pause %*
set "run_exit_code=%ERRORLEVEL%"
endlocal & exit /b %run_exit_code%
