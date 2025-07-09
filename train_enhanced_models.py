"""
Enhanced NBA Model Training Script

This script trains improved ensemble models with enhanced features, feature selection,
model calibration, and better hyperparameters for higher accuracy.
"""

import sys
import os
import json
import numpy as np
import config
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


# Import XGBoost for hyperparameter tuning
try:
    import xgboost as xgb
except ImportError:
    print("Warning: XGBoost not available for hyperparameter tuning")
    xgb = None

# Add necessary paths to system path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import custom modules
from utils.shared_utils import select_best_features, get_team_stats, create_comparison_features
from models.ensemble_model import EnsembleNBAPredictor
from models.advanced_ensemble import AdvancedEnsembleNBAPredictor

def hyperparameter_tuning(X, y, model_type='basic'):
    """
    Perform hyperparameter tuning for better accuracy.
    
    Args:
        X (np.array): Training features
        y (np.array): Training labels
        model_type (str): Type of model to tune
    
    Returns:
        dict: Best hyperparameters
    """
    print(f"Performing hyperparameter tuning for {model_type} ensemble...")
    
    if model_type == 'basic':
        # Check if XGBoost is available
        if xgb is None:
            print("Warning: XGBoost not available, skipping hyperparameter tuning")
            return {}
        
        # Tune XGBoost parameters
        xgb_params = {
            'n_estimators': [800, 1200, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [6, 7, 8],
            'subsample': [0.8, 0.85, 0.9],
            'colsample_bytree': [0.8, 0.85, 0.9]
        }
        
        try:
            xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            
            grid_search = GridSearchCV(
                xgb_model, xgb_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)
            
            print(f"Best XGBoost parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_params_
        except Exception as e:
            print(f"Error during hyperparameter tuning: {e}")
            return {}
    
    return {}

def train_enhanced_models(season_year=None, model_type='both', tune_hyperparameters=False):
    """
    Train enhanced models with all improvements.
    
    Args:
        season_year (int): Season year for training data
        model_type (str): Type of model to train ('basic', 'advanced', 'both')
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning
    """
    if season_year is None:
        season_year = config.SEASON_YEAR
    
    print(f"Training enhanced models for {season_year} season...")
    print("=" * 60)
    
    # Train basic ensemble
    if model_type in ['basic', 'both']:
        print("\n🔄 Training Enhanced Basic Ensemble...")
        try:
            basic_model = EnsembleNBAPredictor()
            
            # Load data
            X, y = basic_model.load_data()
            print(f"Loaded {len(X)} training samples with {X.shape[1]} features")
            
            # Apply feature selection
            print("Applying feature selection...")
            X_selected, selected_indices, feature_scores = select_best_features(
                X, y, method='mutual_info'
            )
            basic_model.X = X_selected
            basic_model.feature_names = [basic_model.feature_names[i] for i in selected_indices]
            basic_model.selected_feature_indices = selected_indices
            
            print(f"Selected {len(basic_model.feature_names)} best features")
            
            # Hyperparameter tuning if requested
            if tune_hyperparameters:
                best_params = hyperparameter_tuning(X_selected, y, 'basic')
                # Update model parameters with best found
                if best_params:
                    basic_model.models['xgboost'].set_params(**best_params)
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Initialize and train models
            basic_model.initialize_models()
            basic_model.train_models(X_train, y_train, X_test, y_test)
            
            # Evaluate performance on full dataset
            y_pred_full = basic_model.predict(X_selected)
            full_accuracy = accuracy_score(y, y_pred_full)
            print(f"Basic Ensemble Full Dataset Accuracy: {full_accuracy:.4f}")
            
            # Test set evaluation
            y_pred = basic_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Basic Ensemble Test Set Accuracy: {test_accuracy:.4f}")
            
            # Save model
            weights_file = f'data/{season_year}_ensemble_basic_weights.json'
            basic_model.save_ensemble(weights_file)
            print(f"✅ Basic ensemble saved to {weights_file}")
            
        except Exception as e:
            print(f"❌ Error training basic ensemble: {e}")
    
    # Train advanced ensemble
    if model_type in ['advanced', 'both']:
        print("\n🔄 Training Enhanced Advanced Ensemble...")
        try:
            advanced_model = AdvancedEnsembleNBAPredictor()
            
            # Load data
            X, y = advanced_model.load_data()
            print(f"Loaded {len(X)} training samples with {X.shape[1]} features")
            
            # Hyperparameter tuning if requested
            if tune_hyperparameters:
                best_params = hyperparameter_tuning(X, y, 'advanced')
                # Update model parameters with best found
                if best_params:
                    advanced_model.models['xgboost'].set_params(**best_params)
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Initialize models
            advanced_model.initialize_models()
            
            # Train stacking ensemble with better cross-validation
            advanced_model.train_stacking_ensemble(X_train, y_train, X_test, y_test)
            
            # Train voting ensemble
            advanced_model.train_voting_ensemble(X_train, y_train)
            
            # Optimize betting thresholds if odds data available
            if hasattr(advanced_model, 'odds') and len(advanced_model.odds) > 0:
                odds_train = [advanced_model.odds[i] for i in range(len(X)) if i < len(X_train)]
                advanced_model.optimize_betting_thresholds(X_train, y_train, odds_train)
            
            # Evaluate performance on full dataset
            y_pred_full = advanced_model.predict(X)
            full_accuracy = accuracy_score(y, y_pred_full)
            print(f"Advanced Ensemble Full Dataset Accuracy: {full_accuracy:.4f}")
            
            # Test set evaluation
            y_pred = advanced_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Advanced Ensemble Test Set Accuracy: {test_accuracy:.4f}")
            
            # Save model
            weights_file = f'data/{season_year}_ensemble_advanced_weights.json'
            advanced_model.save_ensemble(weights_file)
            print(f"✅ Advanced ensemble saved to {weights_file}")
            
        except Exception as e:
            print(f"❌ Error training advanced ensemble: {e}")
    
    print("\n🎉 Enhanced model training completed!")

def evaluate_model_performance(season_year=None):
    """
    Evaluate the performance of trained models.
    
    Args:
        season_year (int): Season year for evaluation
    """
    if season_year is None:
        season_year = config.SEASON_YEAR
    
    print(f"\n📊 Evaluating Model Performance for {season_year} season...")
    print("=" * 60)
    
    # Load data once for both evaluations
    print("Loading evaluation data...")
    from utils.shared_utils import get_team_stats
    import json
    
    # Load the same data that was used for training
    data_file = f'data/{season_year}-season.json'
    if not os.path.exists(data_file):
        data_file = 'data/2024-season.json'  # Fallback
    
    with open(data_file, 'r') as f:
        games_data = json.load(f)
    
    X_eval = []
    y_eval = []
    
    for game_key, game_data in games_data.items():
        if isinstance(game_data, dict) and 'result' in game_data:
            team_keys = [key for key in game_data.keys() 
                       if key not in ['result', 'home_odds', 'away_odds']]
            
            if len(team_keys) == 2:
                away_team = team_keys[0]
                home_team = team_keys[1]
                
                away_stats = game_data[away_team]
                home_stats = game_data[home_team]
                
                features, _ = create_comparison_features(away_stats, home_stats)
                if features is not None:
                    X_eval.append(features)
                    y_eval.append(game_data['result'])
    
    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)
    print(f"Loaded {len(X_eval)} evaluation samples with {X_eval.shape[1]} features")
    
    # Evaluate basic ensemble
    try:
        weights_file = f'data/{season_year}_ensemble_basic_weights.json'
        
        if os.path.exists(weights_file):
            print("Loading basic ensemble...")
            basic_model = EnsembleNBAPredictor()
            basic_model.load_ensemble(weights_file)
            
            # Apply feature selection (after scaling, like during training)
            X_basic = X_eval.copy()
            if hasattr(basic_model, 'selected_feature_indices') and basic_model.selected_feature_indices is not None:
                X_basic = X_eval[:, basic_model.selected_feature_indices]
                print(f"Applied feature selection: {X_basic.shape[1]} features")
            
            # Evaluate on full dataset (no cross-validation since models are pre-trained)
            y_pred = basic_model.predict(X_basic)
            full_accuracy = accuracy_score(y_eval, y_pred)
            print(f"Basic Ensemble Full Dataset Accuracy: {full_accuracy:.4f}")
            
            # Test set evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_basic, y_eval, test_size=0.2, random_state=42, stratify=y_eval
            )
            y_pred = basic_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Basic Ensemble Test Set Accuracy: {test_accuracy:.4f}")
            
        else:
            print("❌ Basic ensemble weights not found")
            
    except Exception as e:
        print(f"❌ Error evaluating basic ensemble: {e}")
    
    # Evaluate advanced ensemble
    try:
        weights_file = f'data/{season_year}_ensemble_advanced_weights.json'
        
        if os.path.exists(weights_file):
            print("Loading advanced ensemble...")
            advanced_model = AdvancedEnsembleNBAPredictor()
            advanced_model.load_ensemble(weights_file)
            
            # Apply feature selection (after scaling, like during training)
            X_advanced = X_eval.copy()
            if hasattr(advanced_model, 'selected_feature_indices') and advanced_model.selected_feature_indices is not None:
                X_advanced = X_eval[:, advanced_model.selected_feature_indices]
                print(f"Applied feature selection: {X_advanced.shape[1]} features")
            
            # Evaluate on full dataset (no cross-validation since models are pre-trained)
            y_pred = advanced_model.predict(X_advanced)
            full_accuracy = accuracy_score(y_eval, y_pred)
            print(f"Advanced Ensemble Full Dataset Accuracy: {full_accuracy:.4f}")
            
            # Test set evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_advanced, y_eval, test_size=0.2, random_state=42, stratify=y_eval
            )
            y_pred = advanced_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f"Advanced Ensemble Test Set Accuracy: {test_accuracy:.4f}")
            
        else:
            print("❌ Advanced ensemble weights not found")
            
    except Exception as e:
        print(f"❌ Error evaluating advanced ensemble: {e}")

def main():
    """Main function to run enhanced training"""
    if len(sys.argv) < 2:
        print("Usage: python train_enhanced_models.py <season_year> [model_type] [tune_hyperparameters]")
        print("Example: python train_enhanced_models.py 2025 both true")
        print("\nModel types: basic, advanced, both")
        print("Season years: 2024, 2025, etc.")
        print("tune_hyperparameters: true/false (default: false)")
        return
    
    try:
        season_year = int(sys.argv[1])
    except ValueError:
        print("Error: Season year must be a valid integer")
        return
    
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'both'
    tune_hyperparameters = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    
    if model_type not in ['basic', 'advanced', 'both']:
        print("Error: Model type must be 'basic', 'advanced', or 'both'")
        return
    
    print(f"Training with hyperparameter tuning: {tune_hyperparameters}")
    
    # Train models
    train_enhanced_models(season_year, model_type, tune_hyperparameters)
    
    # Evaluate performance
    evaluate_model_performance(season_year)
    
    print(f"\n✅ Enhanced training completed for {season_year} season!")

if __name__ == "__main__":
    main() 