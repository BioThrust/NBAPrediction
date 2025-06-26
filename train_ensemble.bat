@echo off
echo ========================================
echo NBA Prediction - Ensemble Model Training
echo ========================================
echo.

echo Starting ensemble model training...
echo This will train both basic and advanced ensemble models:
echo.
echo Basic Ensemble:
echo - Combines XGBoost, Random Forest, Logistic Regression, and Neural Network
echo - Uses weighted voting based on cross-validation performance
echo - Improved accuracy over single models
echo.
echo Advanced Ensemble:
echo - Stacking ensemble with meta-learner
echo - Voting ensemble with multiple strategies
echo - Kelly Criterion betting analysis
echo - Confidence intervals and ROI calculations
echo.

echo Note: This process may take 5-10 minutes depending on your system.
echo Make sure you have run collect_data.bat first to generate the dataset.
echo.

cd ensemble_models

echo Running ensemble training...
python run_ensemble.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Ensemble training completed successfully!
    echo ========================================
    echo.
    echo You can now run run_prediction.bat to make predictions with ensemble models.
    echo.
    echo Files created:
    echo - Ensemble model weights and configurations
    echo - Training performance metrics
    echo.
    echo Tip: Use option 2 (Basic Ensemble) or option 3 (Advanced Ensemble) 
    echo when running predictions for best results.
) else (
    echo.
    echo ========================================
    echo Error: Ensemble training failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo Make sure you have:
    echo 1. Run collect_data.bat first to generate the dataset
    echo 2. Run train_model.bat to create the neural network weights
    echo 3. All required dependencies installed (pip install -r ensemble_models/requirements_ensemble.txt)
    echo 4. Sufficient disk space and memory
)

cd ..

echo.
echo Press any key to exit...
pause >nul 