@echo off
title ASL Speller - Setup
color 0A

echo ============================================
echo        ASL Speller - Auto Setup
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)

echo [1/4] Python found:
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo [2/4] Creating virtual environment...
    python -m venv .venv
) else (
    echo [2/4] Virtual environment already exists, skipping...
)
echo.

:: Activate venv and install packages
echo [3/4] Installing packages into .venv...
call .venv\Scripts\activate
pip install --upgrade pip --quiet
pip install mediapipe opencv-python scikit-learn numpy torch --index-url https://download.pytorch.org/whl/cpu
echo.

:: Download hand_landmarker.task if not present
if not exist "hand_landmarker.task" (
    echo [4/4] Downloading hand_landmarker.task model...
    curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
) else (
    echo [4/4] hand_landmarker.task already exists, skipping...
)
echo.

echo ============================================
echo   Setup complete! You can now run:
echo.
echo   1. python collect_data.py
echo   2. python train_models.py
echo   3. python asl_speller.py
echo ============================================
echo.
pause
