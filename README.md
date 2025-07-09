# NBA Game Prediction System

A sophisticated machine learning system for predicting NBA game outcomes using ensemble models, advanced feature engineering, and interactive betting simulation.

Note: This predicts who is ***more likely*** to win, and thus will get many results wrong 

## 🏀 Features

### **Multiple Prediction Models**
- **Neural Network**: Original deep learning model
- **Basic Ensemble**: Fast, accurate model using weighted averaging of multiple algorithms
- **Advanced Ensemble**: Sophisticated model with stacking, voting, confidence intervals, and betting analysis

### **Interactive Betting Interface**
- Manual betting on individual games
- Real-time balance tracking
- ROI and accuracy statistics
- Odds-based profit/loss calculations

### **Advanced Analytics**
- 60+ sophisticated features including team stats, player performance, and matchup analysis
- Feature selection and scaling
- Confidence intervals and betting recommendations
- Cross-validation and hyperparameter tuning

## 🚀 Quick Start

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements_enhanced.txt
```

### Training Models
```bash
# Train basic ensemble model
train_ensemble.bat

# Train advanced ensemble model  
train_enhanced.bat

# Train neural network model
train_neural.bat
```

### Running Predictions
```bash
# Testing it against actual results (includes simulation betting interface)
test_results.bat

# Single game prediction
predict_game.py

# Testing it against betting odds (who is more likely to win)
test_prediction.bat
```

## 📊 Model Comparison

| Model Type | Speed | Accuracy | Features |
|------------|-------|----------|----------|
| **Neural Network** | Medium | Good | Basic features |
| **Basic Ensemble** | Fast | Very Good | 60+ features, weighted averaging |
| **Advanced Ensemble** | Slower | Excellent | 60+ features, stacking, voting, betting analysis |

## 🎯 Advanced Ensemble Features

### **Model Stacking**
- XGBoost, LightGBM, Random Forest, SVM, Logistic Regression
- Meta-learners for optimal combination
- Cross-validation for robust performance

### **Betting Analysis**
- Confidence-based betting thresholds
- Risk-adjusted recommendations
- Historical performance tracking

### **Feature Engineering**
- Team offensive/defensive ratings
- Pace, efficiency metrics
- Head-to-head matchup analysis
- Player injury considerations

## 📁 Project Structure

```
NBAPrediction/
├── data/                          # Data files
│   ├── 2025-season.json          # Season data with odds
│   ├── *_ensemble_*_weights.json # Model weights
│   └── *_team_stats_cache.json   # Cached team stats
├── models/                        # Model implementations
│   ├── ensemble_model.py         # Basic ensemble
│   ├── advanced_ensemble.py      # Advanced ensemble
│   └── sports_binary.py          # Neural network
├── data_collection/              # Data scraping
│   ├── basketball_reference_scraper/
│   └── data_scraper_main.py
├── utils/                        # Shared utilities
├── *.bat                         # Windows batch files
└── requirements_*.txt            # Dependencies
```

## 🎮 Usage Examples

### Interactive Betting
```bash
# Run betting interface
test_results.bat

# Choose model and start betting
# 1. Select Advanced Ensemble
# 2. Choose Interactive betting
# 3. Place bets on games with odds
```

### Single Game Prediction
```python
from models.advanced_ensemble import AdvancedEnsembleNBAPredictor

# Load model
model = AdvancedEnsembleNBAPredictor()
model.load_weights('data/2025_ensemble_advanced_weights.json')

# Predict game
result = model.predict_game_advanced('BOS', 'LAL')
print(f"Prediction: {result['ensemble_prediction']}")
print(f"Confidence: {result['ensemble_probability']:.1%}")
```

### Batch Testing
```bash
# Test model accuracy on season data
test_results.bat
# Choose "Prediction accuracy only"
```

## 🔧 Configuration

### Model Parameters
- **Basic Ensemble**: 5 models, weighted averaging
- **Advanced Ensemble**: 11+ models, stacking with meta-learners
- **Feature Selection**: Automatic selection of most predictive features
- **Scaling**: Standardization for optimal model performance

### Betting Settings
- **Confidence Thresholds**: Adjustable for different risk levels
- **Odds Integration**: Real odds data from season files
- **Balance Tracking**: Real-time profit/loss calculation

## 📈 Performance

### Model Accuracy (2025 Season)
- **Neural Network**: ~65-70%
- **Basic Ensemble**: ~72-75%
- **Advanced Ensemble**: ~75-78%

### Betting Performance
- **ROI**: Varies by strategy and confidence thresholds
- **Win Rate**: Typically 60-70% on high-confidence predictions
- **Risk Management**: Built-in confidence thresholds

## 🛠️ Development

### Adding New Features
```python
# In utils/shared_utils.py
def create_comparison_features(away_stats, home_stats):
    # Add new feature calculations here
    features = []
    # ... feature engineering
    return features
```

### Training New Models
```python
# In models/advanced_ensemble.py
class AdvancedEnsembleNBAPredictor:
    def __init__(self):
        # Add new model types here
        self.models = {
            'new_model': NewModelClass()
        }
```

## 📋 Requirements

### Core Dependencies
- `scikit-learn>=1.0`
- `xgboost>=1.5`
- `lightgbm>=3.3`
- `pandas>=1.3`
- `numpy>=1.21`
- `requests>=2.25`
- `beautifulsoup4>=4.9`

### Optional Dependencies
- `selenium` (for advanced scraping)
- `matplotlib` (for visualizations)
- `seaborn` (for enhanced plots)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Basketball Reference for game data
- NBA.com for official statistics
- Scikit-learn community for ML algorithms
- XGBoost and LightGBM teams for gradient boosting implementations

---

**Note**: This system is for educational and research purposes. Betting involves risk and should be done responsibly. 
