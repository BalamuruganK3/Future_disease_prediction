#!/bin/bash
# ============================================================
#  Future Disease Prediction System - Setup & Run Script
#  Run this once to install, train, and launch the web app
# ============================================================

echo "======================================"
echo "  Future Disease Prediction System"
echo "======================================"

# 1. Install dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -q tensorflow flask pandas numpy scikit-learn

# 2. Generate dataset
echo ""
echo "[2/4] Generating training dataset (10,000 samples)..."
python generate_dataset.py

# 3. Train CNN model
echo ""
echo "[3/4] Training CNN model (this takes 2-5 minutes)..."
python train_model.py

# 4. Launch web app
echo ""
echo "[4/4] Starting web application..."
echo ""
echo "  ✅ Open your browser at: http://localhost:5000"
echo "  Press Ctrl+C to stop"
echo ""
python app.py
