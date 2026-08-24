@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

if "%GEMINI_API_KEY%"=="" (
    echo GEMINI_API_KEY is not set.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python -c "import rag; rag.ensure_embeddings(force=True)"

if errorlevel 1 (
    echo Failed to rebuild embeddings.
    pause
    exit /b 1
)

echo Embeddings rebuilt successfully.
pause
