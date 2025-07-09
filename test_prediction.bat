@echo off
echo ========================================
echo NBA Prediction Test Script
echo ========================================
echo.

REM Get season file from user
set /p season_file="Enter season file (e.g., 2025-season.json): "

REM Check if file exists in data folder first, then current directory
if exist "data\%season_file%" (
    set season_file=data\%season_file%
) else if exist "%season_file%" (
    REM File exists in current directory
) else (
    echo Error: Season file "%season_file%" not found!
    echo Please make sure the file exists in the data folder or current directory.
    echo Available files in data folder:
    dir data\*.json /b 2>nul
    pause
    exit /b 1
)

REM Get model type from user
echo.
echo Select model type:
echo 1. Basic Ensemble
echo 2. Advanced Ensemble
set /p model_choice="Enter choice (1 or 2): "

if "%model_choice%"=="1" (
    set model_type=basic
) else if "%model_choice%"=="2" (
    set model_type=advanced
) else (
    echo Invalid choice. Using Advanced Ensemble by default.
    set model_type=advanced
)

REM Get season year from user
echo.
set /p season_year="Enter season year (e.g., 2025): "

REM Run the test prediction script
echo.
echo Running test prediction...
echo Season file: %season_file%
echo Model type: %model_type%
echo Season year: %season_year%
echo.

python test_prediction.py "%season_file%" %model_type% %season_year%

echo.
echo Test completed!
pause 