@echo off
title DeepShield - AI Deepfake Detection
color 0A
echo.
echo  ==========================================
echo   DeepShield -- AI Deepfake Detection
echo  ==========================================
echo.
echo  [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install Python 3.8+ from python.org
    pause
    exit /b
)
echo  [OK] Python found.

echo.
echo  [2/3] Installing dependencies (first run may take a minute)...
pip install flask flask-cors numpy librosa soundfile pillow opencv-python-headless scikit-learn scipy mysql-connector-python --quiet --break-system-packages 2>nul || pip install flask flask-cors numpy librosa soundfile pillow opencv-python-headless scikit-learn scipy mysql-connector-python --quiet

echo  [OK] Dependencies ready.
echo.
echo  [3/3] Starting server...
echo.
echo  ==========================================
echo   Open your browser at: http://127.0.0.1:5000
echo  ==========================================
echo.

cd /d "%~dp0"
python app.py

pause
