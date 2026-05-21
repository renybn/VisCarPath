@echo off
echo ========================================
echo  VisCarPath Environment Repair
echo ========================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo Creating fresh virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)

echo Installing/Updating all dependencies...
pip install -r requirements.txt --upgrade

echo.
echo Environment ready! Close this window and run `run.bat`.
pause