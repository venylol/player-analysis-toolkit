@echo off
chcp 65001 >nul
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 "C:\path\to\extracted_v3_package"
  exit /b 2
)
set "SERVER_ROOT=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_incremental_merge_and_data_prep.ps1" -ServerRoot "%SERVER_ROOT%"
exit /b %ERRORLEVEL%
