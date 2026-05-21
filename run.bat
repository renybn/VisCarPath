@echo off
echo ========================================
echo  VisCarPath Navigation Launcher
echo ========================================
echo

REM 1. Check/Create venv if missing
if not exist "venv\Scripts\activate.bat" (
    echo First run detected. Creating virtual environment...
    python -m venv venv
)

REM 2. Activate environment
echo 🔌 Activating environment...
call venv\Scripts\activate.bat

REM 3. Verify & install dependencies
echo 📥 Installing/Updating dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Starting Autonomous Navigation...
echo ========================================
python main_navigation.py --target 0
echo.
echo Navigation stopped.
pause