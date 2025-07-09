@echo off
setlocal enabledelayedexpansion

REM ========================================
REM NBA Prediction - Enhanced Game Predictor
REM ========================================
echo ========================================
echo NBA Prediction - Enhanced Game Predictor
echo ========================================
echo.

echo Usage: run_prediction.bat [AWAY_TEAM] [HOME_TEAM] [MODEL_TYPE] [SEASON_YEAR]
echo Example: run_prediction.bat BOS LAL advanced 2025
echo.
echo Model types: basic, advanced (default: advanced)
echo Season years: 2024, 2025, etc. (default: 2025)
echo.

REM Set model type and season year once at the beginning
if "%3"=="" (
    set /p model_type="Enter model type (basic/advanced, default advanced): "
    if "!model_type!"=="" set model_type=advanced
) else (
    set model_type=%3
)
if "%4"=="" (
    set /p season_year="Enter season year (e.g. 2025, default 2025): "
    if "!season_year!"=="" set season_year=2025
) else (
    set season_year=%4
)

echo.
echo Using model: !model_type!, season: !season_year!
echo.

:predict_loop
REM Prompt for teams (only teams, not model/season)
if "%1"=="" (
    set /p away_team="Enter AWAY team abbreviation (e.g. BOS): "
) else (
    set away_team=%1
    set 1=
)
if "%2"=="" (
    set /p home_team="Enter HOME team abbreviation (e.g. LAL): "
) else (
    set home_team=%2
    set 2=
)

echo.
echo Running enhanced prediction for !away_team! @ !home_team! (!model_type!, !season_year!)
echo.
python predict_game.py !away_team! !home_team! !model_type! !season_year!
echo.
echo Prediction completed.
echo.
set /p again="Predict another game? (y/n): "
if /i "!again!"=="y" goto predict_loop

echo Exiting NBA Prediction.
pause >nul
endlocal 