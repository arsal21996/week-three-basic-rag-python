@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=py
) else (
    set PYTHON=python
)

echo Creating virtual environment...
%PYTHON% -m venv .venv
if errorlevel 1 goto :error

call .venv\Scripts\activate.bat
if errorlevel 1 goto :error

python -m pip install --upgrade pip
if errorlevel 1 goto :error

python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete.
echo Next: edit run.bat and set your OPENAI_API_KEY, then run run.bat.
pause
exit /b 0

:error
echo.
echo Setup failed. Make sure Python 3.10+ is installed and available on PATH.
pause
exit /b 1
