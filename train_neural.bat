@echo off
echo ========================================
echo NBA Prediction - Model Training
echo ========================================
echo.

echo Starting neural network training...
echo This will:
echo 1. Load the 2024 season dataset
echo 2. Normalize features for training
echo 3. Perform 5-fold cross-validation
echo 4. Train the neural network with early stopping
echo 5. Save the trained weights to json_files/weights.json
echo.

echo Note: This process may take several minutes depending on your system.
echo.

cd data_collection

echo Running sports_binary.py...
python sports_binary.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Model training completed successfully!
    echo ========================================
    echo.
    echo Files created:
    echo - json_files/weights.json (trained neural network weights)
    echo.
    echo Model performance metrics have been displayed above.
    echo You can now run predict_game.py to make predictions.
    echo.
    echo Optional: You can also run ensemble training:
    echo cd ensemble_models
    echo python run_ensemble.py
) else (
    echo.
    echo ========================================
    echo Error: Model training failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo Make sure you have:
    echo 1. Run collect_data.bat first to generate the dataset
    echo 2. All required dependencies installed
    echo 3. Sufficient disk space for the model weights
)

cd ..

echo.
echo Press any key to exit...
pause >nul 