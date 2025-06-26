@echo off
echo ========================================
echo NBA Prediction - Data Collection
echo ========================================
echo.

echo Starting data collection process...
echo This will:
echo 1. Scrape team statistics from Basketball Reference
echo 2. Collect historical odds data from OddsPortal
echo 3. Generate the 2024 season dataset
echo 4. Cache team statistics for faster future runs
echo.

echo Note: This process may take several minutes depending on your internet connection.
echo.

cd data_collection

echo Running playoff_data.py...
python playoff_data.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Data collection completed successfully!
    echo ========================================
    echo.
    echo Files created:
    echo - json_files/2024-season.json (main dataset)
    echo - json_files/team_stats_cache.json (cached team stats)
    echo.
    echo You can now run train_model.bat to train the neural network.
) else (
    echo.
    echo ========================================
    echo Error: Data collection failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo Make sure you have all required dependencies installed.
)

cd ..

echo.
echo Press any key to exit...
pause >nul 