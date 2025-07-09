@echo off
setlocal enabledelayedexpansion

echo Training Enhanced NBA Prediction Models...
echo.

REM Check if season year is provided
if "%1"=="" (
    echo Usage: train_enhanced.bat [season_year] [model_type] [tune_hyperparameters]
    echo Example: train_enhanced.bat 2025 both true
    echo.
    echo Model types: basic, advanced, both (default: both)
    echo Season years: 2024, 2025, etc. (default: 2025)
    echo tune_hyperparameters: true/false (default: false)
    echo.
    set /p season_year="Enter season year (default 2025): "
    echo Debug: season_year after input = "!season_year!"
    if "!season_year!"=="" (
        set season_year=2025
        echo Debug: Set season_year to default = "!season_year!"
    )
    set /p model_type="Enter model type (basic/advanced/both, default both): "
    echo Debug: model_type after input = "!model_type!"
    if "!model_type!"=="" (
        set model_type=both
        echo Debug: Set model_type to default = "!model_type!"
    )
    set /p tune_hyperparams="Enable hyperparameter tuning? (true/false, default false): "
    echo Debug: tune_hyperparams after input = "!tune_hyperparams!"
    if "!tune_hyperparams!"=="" (
        set tune_hyperparams=false
        echo Debug: Set tune_hyperparams to default = "!tune_hyperparams!"
    )
) 

echo.
echo Training enhanced models for !season_year! season...
echo Model type: !model_type!
echo Hyperparameter tuning: !tune_hyperparams!
echo.

REM Debug: Show what we're passing to Python
echo Debug: Passing to Python: "!season_year!" "!model_type!" "!tune_hyperparams!"
echo.

REM Call Python script with proper argument passing
python train_enhanced_models.py "!season_year!" "!model_type!" "!tune_hyperparams!"

echo.
echo Training completed!
pause
endlocal 