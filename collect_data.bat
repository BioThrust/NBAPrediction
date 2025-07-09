@echo off
setlocal enabledelayedexpansion
echo ========================================
echo NBA Prediction - Data Collection
echo ========================================
echo.

echo Choose data collection mode:
echo 1. Single season (e.g., 2024 for 2023-2024 season)
echo 2. Multiple seasons (e.g., 2022,2023,2024 for multiple years)
echo.
set /p MODE="Enter 1 or 2: "

if "%MODE%"=="1" goto single
if "%MODE%"=="2" goto multiple
echo Invalid choice. Please run the script again and select 1 or 2.
goto end

:single
set /p SEASON_YEAR="Enter the NBA ending season year (e.g., 2024 for 2023-2024 season): "
echo.
echo Starting data collection process for %SEASON_YEAR% season...
echo This will:
echo 1. Scrape team statistics from Basketball Reference
echo 2. Collect historical odds data from OddsPortal
echo 3. Generate the %SEASON_YEAR% season dataset
echo 4. Cache team statistics for faster future runs
echo.

echo Running data_scraper_main.py for %SEASON_YEAR% season...
python -m data_collection.data_scraper_main %SEASON_YEAR%
set PYTHON_EXIT_CODE=%ERRORLEVEL%

echo.
echo DEBUG: Python script finished with exit code: %PYTHON_EXIT_CODE%

REM Add a small delay to ensure ERRORLEVEL is properly set
timeout /t 1 /nobreak >nul

echo DEBUG: After delay, exit code is: %PYTHON_EXIT_CODE%

if %PYTHON_EXIT_CODE% EQU 0 (
    echo.
    echo ========================================
    echo Data collection completed successfully!
    echo ========================================
    echo.
    echo Files created:
    if exist "data\%SEASON_YEAR%-season.json" (
        echo - data/%SEASON_YEAR%-season.json ✓
    ) else (
        echo - data/%SEASON_YEAR%-season.json ✗ (file not found)
    )
    if exist "data\%SEASON_YEAR%_team_stats_cache.json" (
        echo - data/team_stats_cache_%SEASON_YEAR%.json ✓
    ) else (
        echo - data/team_stats_cache_%SEASON_YEAR%.json ✗ (file not found)
    )
    echo.
    echo You can now run train_model.bat to train the neural network.
) 
if %PYTHON_EXIT_CODE% NEQ 0 (
    echo.
    echo ========================================
    echo Error: Data collection failed!
    echo ========================================
    echo.
    echo Python script exited with code: %PYTHON_EXIT_CODE%
    echo Please check the error messages above and try again.
    echo Make sure you have all required dependencies installed.
)
goto end

:multiple
set /p SEASONS="Enter multiple season years separated by commas (e.g., 2022,2023,2024): "
echo.
echo Starting data collection process for multiple seasons: %SEASONS%
echo This will:
echo 1. Scrape team statistics from Basketball Reference for each season
echo 2. Collect historical odds data from OddsPortal for each season
echo 3. Generate separate datasets for each season
echo 4. Cache team statistics for faster future runs
echo.

set SEASONS_PARSED=%SEASONS:,= %

for %%s in (%SEASONS_PARSED%) do (
    echo.
    echo ========================================
    echo Processing season %%s...
    echo ========================================
    echo Running data_scraper_main.py for %%s season...
    python -m data_collection.data_scraper_main %%s
    set PYTHON_EXIT_CODE=!ERRORLEVEL!
    
    echo DEBUG: Season %%s finished with exit code: !PYTHON_EXIT_CODE!
    
    REM Add a small delay to ensure ERRORLEVEL is properly set
    timeout /t 1 /nobreak >nul
    
    if !PYTHON_EXIT_CODE! EQU 0 (
        echo Season %%s completed successfully!
    ) else (
        echo Error: Season %%s failed with exit code !PYTHON_EXIT_CODE!
    )
)

echo.
echo ========================================
echo Multi-season data collection completed!
echo ========================================
echo.
echo Files created:
for %%s in (%SEASONS_PARSED%) do (
    if exist "data\%%s-season.json" (
        echo - data/%%s-season.json ✓
    ) else (
        echo - data/%%s-season.json ✗ (failed)
    )
    if exist "data\team_stats_cache_%%s.json" (
        echo - data/team_stats_cache_%%s.json ✓
    ) else (
        echo - data/team_stats_cache_%%s.json ✗ (failed)
    )
)
echo.
echo You can now run train_model.bat to train the neural network.
echo Or run combine_seasons.bat to combine the successful datasets.

goto end

:end
echo.
echo Press any key to exit...
pause >nul 