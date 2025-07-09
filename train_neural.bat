@echo off
echo ========================================
echo NBA Prediction - Model Training
echo ========================================
echo.

echo Choose training data source:
echo 1. Single season (e.g., 2024 for 2023-2024 season)
echo 2. Combined dataset (multiple seasons combined)
echo.
set /p DATA_SOURCE="Enter 1 or 2: "

if "%DATA_SOURCE%"=="1" (
    set /p SEASON_YEAR="Enter the NBA ending season year (e.g., 2024 for 2023-2024 season): "
    echo.
    echo Starting neural network training for %SEASON_YEAR% season...
    echo This will:
    echo 1. Load the %SEASON_YEAR% season dataset
    echo 2. Normalize features for training
    echo 3. Perform 5-fold cross-validation
    echo 4. Train the neural network with early stopping
    echo 5. Save the trained weights to data/weights.json
    echo.
    
    cd data_collection
    echo Running sports_binary.py for %SEASON_YEAR% season...
    python sports_binary.py %SEASON_YEAR%
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ========================================
        echo Model training completed successfully!
        echo ========================================
        echo.
        echo Files created:
        echo - data/weights.json (trained neural network weights)
        echo.
        echo Model performance metrics have been displayed above.
        echo You can now run predict_game.py to make predictions.
        echo.
        echo Optional: You can also run ensemble training:
        echo cd models
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
) else if "%DATA_SOURCE%"=="2" (
    echo.
    echo Starting neural network training on combined dataset...
    echo This will:
    echo 1. Load the combined-seasons.json dataset
    echo 2. Normalize features for training
    echo 3. Perform 5-fold cross-validation
    echo 4. Train the neural network with early stopping
    echo 5. Save the trained weights to data/weights.json
    echo.
    echo Note: Combined dataset training may take longer but should provide better results.
    echo.
    
    cd data_collection
    echo Running sports_binary.py on combined dataset...
    python sports_binary.py combined
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ========================================
        echo Model training completed successfully!
        echo ========================================
        echo.
        echo Files created:
        echo - data/weights.json (trained neural network weights)
        echo.
        echo Model performance metrics have been displayed above.
        echo The model was trained on multiple seasons for better performance.
        echo You can now run predict_game.py to make predictions.
        echo.
        echo Optional: You can also run ensemble training:
        echo cd models
        echo python run_ensemble.py
    ) else (
        echo.
        echo ========================================
        echo Error: Model training failed!
        echo ========================================
        echo.
        echo Please check the error messages above and try again.
        echo Make sure you have:
        echo 1. Run collect_data.bat first to generate individual season datasets
        echo 2. Run combine_seasons.bat to create the combined dataset
        echo 3. All required dependencies installed
        echo 4. Sufficient disk space for the model weights
    )
    
    cd ..
) else (
    echo Invalid choice. Please run the script again and select 1 or 2.
)

echo.
echo Press any key to exit...
pause >nul 