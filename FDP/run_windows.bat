@echo off
echo ======================================
echo   Future Disease Prediction System
echo ======================================

echo.
echo [1/4] Installing Python dependencies...
pip install tensorflow flask pandas numpy scikit-learn

echo.
echo [2/4] Generating training dataset...
python generate_dataset.py

echo.
echo [3/4] Training CNN model (takes 2-5 minutes)...
python train_model.py

echo.
echo [4/4] Starting web application...
echo.
echo   Open your browser at: http://localhost:5000
echo.
python app.py
pause
