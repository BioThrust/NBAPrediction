@echo off
setlocal enabledelayedexpansion
echo ========================================
echo NBA Prediction - Season Data Combiner
echo ========================================
echo.

echo This tool combines multiple season datasets into one large training dataset.
echo This is useful for training models on multiple years of data for better performance.
echo.

set /p SEASONS="Enter season years to combine, separated by commas (e.g., 2022,2023,2024): "

echo.
echo Starting season combination process...
echo This will:
echo 1. Load each season dataset
echo 2. Combine them into one large dataset
echo 3. Save the combined dataset for training
echo.

cd data_collection

echo Running combine_seasons.py...
echo Seasons to combine: %SEASONS%

REM Parse comma-separated seasons and pass as individual arguments
set SEASONS_PARSED=%SEASONS:,= %
python combine_seasons.py %SEASONS_PARSED%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Season combination completed successfully!
    echo ========================================
    echo.
    echo Files created:
    echo - json_files/combined-seasons.json (combined dataset)
    echo.
    echo You can now use this combined dataset for training:
    echo 1. Run train_neural.bat and specify 'combined' as the season
    echo 2. Or modify training scripts to use 'combined-seasons.json'
    echo 3. The combined dataset will provide more training examples
) else (
    echo.
    echo ========================================
    echo Error: Season combination failed!
    echo ========================================
    echo.
    echo Please check the error messages above and try again.
    echo Make sure you have:
    echo 1. Run collect_data.bat first to generate the individual season datasets
    echo 2. All required dependencies installed
)

cd ..

echo.
echo Press any key to exit...
pause >nul 