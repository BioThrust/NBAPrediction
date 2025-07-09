# NBA Ensemble Prediction System

This implementation provides advanced ensemble methods for NBA game prediction, combining multiple machine learning algorithms to improve accuracy and provide betting insights.

## Features

### Basic Ensemble (`ensemble_model.py`)
- **Multiple Algorithms**: XGBoost, Random Forest, Logistic Regression, Neural Network
- **Weighted Voting**: Automatically calculates optimal weights based on cross-validation performance
- **Simple Interface**: Easy-to-use prediction functions
- **Model Persistence**: Save and load trained ensemble models

### Advanced Ensemble (`advanced_ensemble.py`)
- **Stacking**: Meta-learner that combines base model predictions
- **Voting Ensembles**: Both soft and hard voting strategies
- **Kelly Criterion**: Optimal betting strategy with confidence thresholds
- **Feature Scaling**: Standardized features for better model performance
- **Confidence Intervals**: Model agreement and prediction confidence
- **Betting Analysis**: ROI calculations and betting recommendations

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_ensemble.txt
```

2. Ensure you have your training data in `2024-season.json` format

## Usage

### Quick Start

Run the interactive script:
```bash
python run_ensemble.py
```

### Command Line Usage

**Basic Ensemble:**
```bash
python run_ensemble.py basic BOS LAL
```

**Advanced Ensemble with Odds:**
```bash
python run_ensemble.py advanced BOS LAL 1.85 2.05
```

**Run Both Ensembles:**
```bash
python run_ensemble.py both BOS LAL 1.85 2.05
```

### Programmatic Usage

**Basic Ensemble:**
```python
from ensemble_model import EnsembleNBAPredictor

# Initialize and train
ensemble = EnsembleNBAPredictor()
X, y = ensemble.load_data()
ensemble.initialize_models()
ensemble.train_models(X_train, y_train)

# Make prediction
prediction = ensemble.predict_game("BOS", "LAL")
print(f"Prediction: {prediction['ensemble_prediction']}")
```

**Advanced Ensemble:**
```python
from advanced_ensemble import AdvancedEnsembleNBAPredictor

# Initialize and train
ensemble = AdvancedEnsembleNBAPredictor()
X, y = ensemble.load_data()
ensemble.initialize_models()
ensemble.train_stacking_ensemble(X_train, y_train, X_val, y_val)

# Make prediction with odds
odds = {'home': 1.85, 'away': 2.05}
prediction = ensemble.predict_game_advanced("BOS", "LAL", odds)
print(f"Prediction: {prediction['ensemble_prediction']}")
print(f"Betting Recommendations: {prediction['betting_recommendations']}")
```

## Algorithms Included

### 1. XGBoost
- **Best for**: Sports prediction, handles non-linear relationships
- **Features**: Gradient boosting with regularization
- **Typical Performance**: 60-70% accuracy

### 2. Random Forest
- **Best for**: Interpretability, feature importance
- **Features**: Ensemble of decision trees
- **Typical Performance**: 58-68% accuracy

### 3. Gradient Boosting
- **Best for**: Sequential learning, complex patterns
- **Features**: Boosting with gradient descent
- **Typical Performance**: 59-69% accuracy

### 4. Logistic Regression
- **Best for**: Baseline model, interpretability
- **Features**: Linear model with regularization
- **Typical Performance**: 55-65% accuracy

### 5. Ridge Classifier
- **Best for**: Regularized linear classification
- **Features**: L2 regularization
- **Typical Performance**: 55-65% accuracy

### 6. Support Vector Machine
- **Best for**: Non-linear classification
- **Features**: Kernel methods
- **Typical Performance**: 57-67% accuracy

### 7. Neural Network
- **Best for**: Complex non-linear patterns
- **Features**: Multi-layer perceptron
- **Typical Performance**: 58-68% accuracy

## Ensemble Strategies

### 1. Weighted Voting
- Automatically calculates optimal weights based on cross-validation
- Weights models by their individual performance
- Formula: `weight = cv_score / sum(all_cv_scores)`

### 2. Stacking
- Uses base model predictions as features for meta-learner
- Meta-learner learns optimal combination strategy
- Typically improves performance by 2-5%

### 3. Voting Ensembles
- **Soft Voting**: Averages probability predictions
- **Hard Voting**: Majority vote on binary predictions
- Both strategies available for comparison

## Betting Strategy (Kelly Criterion)

### Optimal Betting
- Calculates optimal bet size based on edge and odds
- Formula: `f = (bp - q) / b`
  - `f`: fraction of bankroll to bet
  - `b`: net odds received on bet
  - `p`: probability of winning
  - `q`: probability of losing (1-p)

### Confidence Thresholds
- Only bets when model confidence is high enough
- Uses standard deviation of model predictions as confidence measure
- Default threshold: 0.2 (low std = high confidence)

### Risk Management
- Minimum Kelly fraction: 0.05 (5% of bankroll)
- Maximum bet size: 25% of bankroll (safety limit)
- Only bets when expected value is positive

## Performance Metrics

### Accuracy Metrics
- **Overall Accuracy**: Percentage of correct predictions
- **ROC-AUC**: Area under ROC curve (0.5-1.0)
- **Individual Model Performance**: Compare all models

### Betting Metrics
- **Betting Accuracy**: Accuracy on games where bets were placed
- **Total Profit**: Sum of all betting profits/losses
- **ROI**: Return on investment percentage
- **Kelly Score**: Expected profit per unit bet

## Expected Performance

### Realistic Accuracy Ranges
- **Individual Models**: 55-70%
- **Ensemble Models**: 60-75%
- **With Feature Engineering**: 65-80%

### Factors Affecting Performance
- **Data Quality**: More recent data = better performance
- **Feature Engineering**: Advanced features improve accuracy
- **Market Efficiency**: Less efficient markets = higher edge
- **Sample Size**: More training data = more stable performance

## File Structure

```
├── ensemble_model.py          # Basic ensemble implementation
├── advanced_ensemble.py       # Advanced ensemble with betting
├── run_ensemble.py           # Easy-to-use runner script
├── requirements_ensemble.txt  # Dependencies
├── README_ensemble.md        # This file
└── 2024-season.json         # Training data (your file)
```

## Tips for Best Performance

1. **Feature Engineering**: Add more relevant features (injuries, rest days, etc.)
2. **Regular Retraining**: Update models with new data weekly
3. **Cross-Validation**: Use 5-fold CV for reliable performance estimates
4. **Ensemble Diversity**: Include models with different characteristics
5. **Risk Management**: Never bet more than 5% of bankroll on single game
6. **Market Analysis**: Focus on games with clear edges
7. **Continuous Monitoring**: Track performance and adjust strategies

## Troubleshooting

### Common Issues

1. **Import Errors**: Install all requirements with `pip install -r requirements_ensemble.txt`
2. **Data Format**: Ensure `2024-season.json` has correct format with team stats and results
3. **Memory Issues**: Reduce number of estimators in tree-based models
4. **Slow Training**: Use `n_jobs=-1` for parallel processing where available

### Performance Optimization

1. **Feature Selection**: Remove irrelevant features to reduce noise
2. **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters
3. **Data Quality**: Clean and validate training data
4. **Model Selection**: Focus on best-performing models in ensemble

## Example Output

```
=== Advanced NBA Ensemble Model Training ===
Loading training data...
Loaded 1234 games with 45 features
Initializing advanced ensemble models...
Initialized 7 models
Training stacking ensemble...
Training xgboost...
Training random_forest...
Training gradient_boost...
Training logistic...
Training ridge...
Training svm...
Meta-learner trained
Stacking ensemble trained successfully!

=== Predicting BOS @ LAL ===
Ensemble Prediction: 1 (Probability: 0.723)
Confidence: 0.156
Model Agreement: 0.844

Individual Model Predictions:
  xgboost: 1 (Probability: 0.745)
  random_forest: 1 (Probability: 0.712)
  gradient_boost: 1 (Probability: 0.698)
  logistic: 1 (Probability: 0.701)
  ridge: 1 (Probability: 0.715)
  svm: 1 (Probability: 0.728)

Betting Recommendations:
  Bet on HOME: 0.15 (Confidence: 0.844)

Final Ensemble Accuracy: 0.684
Final Ensemble ROC-AUC: 0.712
```

This ensemble system provides a comprehensive approach to NBA prediction with both accuracy optimization and betting strategy implementation. 