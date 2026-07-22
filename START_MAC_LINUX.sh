#!/bin/bash
echo ""
echo "=========================================="
echo "  DeepShield — AI Deepfake Detection"
echo "=========================================="
echo ""
echo "[1/3] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python3 not found. Install from https://python.org"
    exit 1
fi
echo "[OK] $(python3 --version)"

echo ""
echo "[2/3] Installing dependencies..."
pip3 install flask flask-cors numpy librosa soundfile pillow \
    opencv-python-headless scikit-learn scipy mysql-connector-python \
    --quiet --break-system-packages 2>/dev/null || \
pip3 install flask flask-cors numpy librosa soundfile pillow \
    opencv-python-headless scikit-learn scipy mysql-connector-python --quiet
echo "[OK] Dependencies ready."

echo ""
echo "[3/3] Starting server..."
echo ""
echo "=========================================="
echo "  Open browser at: http://127.0.0.1:5000"
echo "=========================================="
echo ""

cd "$(dirname "$0")"
python3 app.py
