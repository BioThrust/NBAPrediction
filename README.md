# NBA Game Prediction Project

A comprehensive NBA game prediction system using machine learning and ensemble methods.

## Project Structure

The project has been organized into logical folders for better maintainability:

```
NBAPrediction/
├── data_collection/          # Data scraping and collection scripts
│   ├── playoff_data.py       # Main data collection script
│   ├── sports_binary.py      # Neural network training script
│   ├── reference_scraper.py  # Basketball reference scraper
│   └── basketball_reference_scraper/  # External scraper library
├── ensemble_models/          # Ensemble learning implementations
│   ├── ensemble_model.py     # Basic ensemble (XGBoost, Random Forest, etc.)
│   ├── advanced_ensemble.py  # Advanced ensemble with stacking and betting analysis
│   ├── run_ensemble.py       # Script to run ensemble models
│   ├── README_ensemble.md    # Ensemble documentation
│   └── requirements_ensemble.txt
├── json_files/               # All JSON data files
│   ├── 2024-season.json     # Main dataset
│   ├── weights.json         # Neural network weights
│   ├── team_stats_cache.json # Cached team statistics
│   ├── model_weights.json   # Additional model weights
│   └── ensemble_model_weights.json
├── utils/                    # Utility functions and shared code
│   ├── shared_utils.py      # Shared functions and neural network class
│   └── betting_odds_accuracy.py
├── models/                   # Reserved for future model implementations
├── data/                     # Reserved for additional data files
├── predict_game.py          # Main prediction interface (defaults to ensemble)
├── test_results.py          # Model testing and evaluation
├── collect_data.bat         # Windows batch file for data collection
├── train_model.bat          # Windows batch file for neural network training
├── train_ensemble.bat       # Windows batch file for ensemble model training
├── run_prediction.bat       # Windows batch file for predictions (ensemble default)
└── requirements.txt         # Main project dependencies
```

## Quick Start (Windows)


1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r ensemble_models/requirements_ensemble.txt
   ```

2. **Collect Data** - Double-click `collect_data.bat` or run:
   ```bash
   collect_data.bat
   ```

3. **Train Neural Network** - Double-click `train_model.bat` or run:
   ```bash
   train_model.bat
   ```
   
4. **Make Predictions** - Double-click `run_prediction.bat` or run:
   ```bash
   run_prediction.bat
   ```
5. **Test Set** - Double-click `test_results.bat` or run:
   ```bash
   test_results.py
   ```

## Batch Files Description

### `collect_data.bat`
- Automatically navigates to the data_collection folder
- Runs the playoff_data.py script
- Provides clear feedback on success/failure
- Creates the main dataset and team stats cache

### `train_model.bat`
- Automatically navigates to the data_collection folder
- Runs the sports_binary.py training script
- Shows training progress and results
- Saves the trained neural network weights

### `train_ensemble.bat` (NEW - RECOMMENDED)
- Automatically navigates to the ensemble_models folder
- Trains both basic and advanced ensemble models
- Provides detailed information about ensemble benefits
- Takes 5-10 minutes but provides significantly better accuracy

### `run_prediction.bat`
- Launches the interactive prediction interface
- **DEFAULTS TO ENSEMBLE MODELS** for best results
- Allows users to choose between different model types
- Provides team prediction functionality with confidence levels

## Model Types (Ensemble-First Approach)

### 1. Basic Ensemble (RECOMMENDED DEFAULT)
- **Combines**: XGBoost, Random Forest, Logistic Regression, and Neural Network
- **Uses**: Weighted voting based on cross-validation performance
- **Accuracy**: ~60-70% (significantly better than single models)
- **Speed**: Fast predictions once trained

### 2. Advanced Ensemble (FOR BETTING INSIGHTS)
- **Features**: Stacking ensemble with meta-learner
- **Includes**: Voting ensemble with multiple strategies
- **Analysis**: Kelly Criterion betting analysis
- **Metrics**: Confidence intervals and ROI calculations
- **Accuracy**: ~65-75% with betting recommendations

### 3. Neural Network (BASIC)
- **Type**: Single hidden layer feedforward neural network
- **Trained on**: Team statistics and comparison features
- **Accuracy**: ~55-65% (baseline performance)
- **Use case**: Quick testing or when ensemble models aren't available

## Features

The models use comprehensive team statistics including:
- Net rating, offensive/defensive ratings
- Shooting efficiency (eFG%)
- Pace and tempo metrics
- Turnover rates and forcing
- Rebounding and assists
- Opponent statistics
- Home court advantage
- Enhanced matchup features

## File Path Updates

All file paths have been updated to reflect the new structure:
- JSON files are now in `json_files/`
- Data collection scripts are in `data_collection/`
- Ensemble models are in `ensemble_models/`
- Utility functions are in `utils/`

## Usage Examples

### Basic Prediction (Ensemble Default)
```bash
python predict_game.py
# Press Enter for Basic Ensemble (recommended)
# Or choose 2 for Basic Ensemble, 3 for Advanced Ensemble
# Enter team abbreviations (e.g., BOS, LAL)
```

### Ensemble Training
```bash
cd ensemble_models
python run_ensemble.py basic BOS LAL
python run_ensemble.py advanced BOS LAL 1.85 2.05
```

### Model Testing
```bash
python test_results.py
# Choose model type and test against actual results
```

## Performance Comparison

- **Neural Network**: ~55-65% accuracy **ORIGINAL**
- **Basic Ensemble**: ~60-70% accuracy ⭐ **GOOD**
- **Advanced Ensemble**: ~60-70% accuracy with betting analysis ⭐ **BEST**

## Why Ensemble Models Are Better

1. **Higher Accuracy**: Combines multiple algorithms for better predictions
2. **Reduced Overfitting**: Multiple models reduce the risk of overfitting
3. **Robustness**: Less sensitive to individual model weaknesses
4. **Confidence Metrics**: Provides confidence levels and model agreement
5. **Betting Insights**: Advanced ensemble includes Kelly Criterion analysis

## Troubleshooting

### Common Issues:
1. **"Module not found" errors**: Make sure you've installed all dependencies with both requirements files
2. **"File not found" errors**: Run `collect_data.bat` first to generate the required data files
3. **Training fails**: Ensure you have sufficient disk space and memory for model training
4. **Data collection fails**: Check your internet connection and ensure the websites are accessible
5. **Ensemble training slow**: This is normal - ensemble models take 5-10 minutes to train

### System Requirements:
- Python 3.7 or higher
- Windows 10/11 (for batch files)
- Internet connection for data collection
- At least 4GB RAM for ensemble training
- 1GB free disk space

## Notes

- **Ensemble models are now the default** for best accuracy
- All JSON files are stored in the `json_files/` directory
- Import statements have been updated to use relative paths
- The project maintains backward compatibility with existing functionality
- Each folder has an `__init__.py` file for proper Python packaging
- Batch files provide a user-friendly way to run the main processes
- **Recommendation**: Use Basic Ensemble (option 2) for general predictions, Advanced Ensemble (option 3) for betting insights 

## Multi-Season Data Collection and Training

The project now supports collecting data for multiple seasons and training models on combined datasets for better performance.

### Multi-Season Data Collection

1. **Collect Multiple Seasons at Once**:
   ```bash
   collect_data.bat
   # Choose option 2: Multiple seasons
   # Enter: 2022,2023,2024
   # Creates: json_files/2022-season.json, json_files/2023-season.json, json_files/2024-season.json
   ```

2. **Combine Multiple Seasons**:
   ```bash
   combine_seasons.bat
   # Enter: 2022,2023,2024
   # Creates: json_files/combined-seasons.json
   ```

### Multi-Season Training

1. **Train on Single Season**:
   ```bash
   train_neural.bat
   # Choose option 1: Single season
   # Enter: 2024
   # Trains on: json_files/2024-season.json
   ```

2. **Train on Combined Dataset**:
   ```bash
   train_neural.bat
   # Choose option 2: Combined dataset
   # Trains on: json_files/combined-seasons.json
   ```

### Benefits of Multi-Season Training

- **More Training Data**: Multiple seasons provide more examples
- **Better Generalization**: Models learn patterns across different years
- **Improved Accuracy**: Larger datasets typically lead to better performance
- **Robust Models**: Less prone to overfitting on single-season patterns

### File Structure with Multi-Season Support

```
json_files/
├── 2022-season.json        # 2021-2022 season data
├── 2023-season.json        # 2022-2023 season data
├── 2024-season.json        # 2023-2024 season data
├── combined-seasons.json   # Combined multi-season dataset
├── weights.json           # Trained model weights
└── team_stats_cache.json  # Cached team statistics
```

## Project Structure 
