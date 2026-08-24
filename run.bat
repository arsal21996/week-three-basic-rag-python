@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run setup.bat first.
    pause
    exit /b 1
)

if "%GEMINI_API_KEY%"=="" (
    echo GEMINI_API_KEY is not set.
    echo.
    echo PowerShell: $env:GEMINI_API_KEY="your-key"
    echo Command Prompt: set GEMINI_API_KEY=your-key
    echo Then run run.bat again.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python rag.py

if errorlevel 1 (
    echo.
    echo Program exited with an error.
    pause
)
