@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "BACKEND_DIR=%ROOT_DIR%backend"

if exist "%BACKEND_DIR%\venv\Scripts\python.exe" (
  set "PYTHON=%BACKEND_DIR%\venv\Scripts\python.exe"
) else if exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
  set "PYTHON=%BACKEND_DIR%\.venv\Scripts\python.exe"
) else (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    set "PYTHON=%%P"
    goto :found_python
  )
  echo No Python interpreter found. Create a backend virtualenv and install backend\requirements.txt.
  exit /b 1
)

:found_python
echo Starting AI Spreadsheet Analyst at http://localhost:5000
echo Using Python: %PYTHON%
cd /d "%BACKEND_DIR%"
if not defined FLASK_DEBUG set "FLASK_DEBUG=0"
"%PYTHON%" run.py
