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

echo Choose ensemble type:
echo - Enter "basic" for basic ensemble (XGBoost, Random Forest, Logistic Regression, Neural Network)
echo - Enter "advanced" for advanced ensemble (Stacking, Voting, Betting Analysis)
echo - Enter "both" to train both ensemble types
echo.

set /p ENSEMBLE_TYPE="Enter ensemble type (basic/advanced/both): "

echo.
echo Enter the season year to train on:
echo - Enter a year (e.g., 2024, 2025) for a specific season
echo - Enter "combined" for multiple seasons combined
echo - Press Enter for interactive mode
echo.

set /p SEASON_INPUT="Enter season year or 'combined': "

cd models

echo Running ensemble training...
if not "%SEASON_INPUT%"=="" (
    echo Using ensemble type: %ENSEMBLE_TYPE%
    echo Using dataset: %SEASON_INPUT%
    python run_ensemble.py %ENSEMBLE_TYPE% %SEASON_INPUT%
) else (
    echo Running in interactive mode...
    python run_ensemble.py %ENSEMBLE_TYPE%
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Ensemble training completed successfully!
    echo ========================================
    echo.
    echo You can now run run_prediction.bat to make predictions with ensemble models.
    echo.
    echo Files created:
    echo - data/[year]_ensemble_basic_weights.json (basic ensemble weights)
    echo - data/[year]_ensemble_advanced_weights.json (advanced ensemble weights)
    echo - Training performance metrics
    echo.
    echo Tip: Use option 2 (Basic Ensemble) or option 3 (Advanced Ensemble) 
    echo when running predictions for best results.
) 
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo Error: Ensemble training failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo Make sure you have:
    echo 1. Run collect_data.bat first to generate the dataset
    echo 2. Run train_model.bat to create the neural network weights
    echo 3. All required dependencies installed (pip install -r models/requirements_ensemble.txt)
    echo 4. Sufficient disk space and memory
)

cd ..

echo.
echo Press any key to exit...
pause >nul 