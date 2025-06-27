@echo off
echo ========================================
echo NBA Prediction - Model Testing
echo ========================================
echo.

set /p SEASON_YEAR="Enter the NBA ending season year to test (e.g., 2024 for 2023-2024 season): "

echo.
echo Starting model testing for %SEASON_YEAR% season...
echo This will:
echo 1. Load the %SEASON_YEAR% season dataset
echo 2. Test different prediction models
echo 3. Compare predictions with actual results
echo 4. Calculate accuracy and betting performance
echo.

echo Note: This process may take several minutes depending on your system.
echo.

echo Running test_results.py for %SEASON_YEAR% season...
python test_results.py %SEASON_YEAR%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Model testing completed successfully!
    echo ========================================
    echo.
    echo Results have been displayed above.
    echo You can run this again with different models or seasons.
) else (
    echo.
    echo ========================================
    echo Error: Model testing failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo Make sure you have:
    echo 1. Run collect_data.bat first to generate the dataset
    echo 2. Run train_neural.bat to train the models
    echo 3. All required dependencies installed
)

echo.
echo Press any key to exit...
pause >nul 