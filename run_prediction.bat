@echo off
echo ========================================
echo NBA Prediction - Game Predictor
echo ========================================
echo.

echo Starting NBA game prediction interface...
echo.
echo Available Models:
echo 1. Neural Network (Original) - Basic single model
echo 2. Basic Ensemble (RECOMMENDED) - Combines multiple algorithms
echo 3. Advanced Ensemble - Includes betting analysis and confidence intervals
echo.

echo RECOMMENDED: Choose option 2 (Basic Ensemble) for best accuracy
echo or option 3 (Advanced Ensemble) for betting insights.
echo.

echo Note: Make sure you have trained models available:
echo - Neural Network: json_files/weights.json
echo - Ensemble models: Will be trained on-demand (may take a few minutes)
echo.

echo Starting prediction interface...
echo When prompted, we recommend selecting option 2 or 3 for best results.
echo.
python predict_game.py

echo.
echo Prediction session ended.
echo Press any key to exit...
pause >nul 