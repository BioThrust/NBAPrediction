"""
Ensemble NBA Prediction Model

This module implements an ensemble prediction model that combines multiple machine learning
algorithms to predict NBA game outcomes. It includes XGBoost, Random Forest, Logistic Regression,
and Neural Network models with weighted voting.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import sys
import os
from sklearn.linear_model import RidgeClassifier
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.shared_utils import get_team_stats, create_comparison_features, PredictionNeuralNetwork
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class EnsembleNBAPredictor:
    """
    Ensemble model that combines multiple machine learning algorithms for NBA game prediction.
    
    This class implements a weighted ensemble of XGBoost, Random Forest, Logistic Regression,
    and Neural Network models to improve prediction accuracy and robustness.
    """
    
    def __init__(self, *args, **kwargs):
        self.model_type = 'ensemble'
        """
        Initialize ensemble model with multiple algorithms.
        """
        self.models = {}
        self.weights = {}
        self.feature_names = None
        self.is_trained = False
    
    def get_data_file_path(self, filename):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, 'data', filename)
    
    def load_data(self, data_file=None):
        """
        Load and prepare training data from JSON file.
        
        Args:
            data_file (str): Path to the JSON file containing game data
        
        Returns:
            tuple: (X, y) - Feature matrix and target labels
        """
        if data_file is None:
            import sys
            if len(sys.argv) > 2:
                try:
                    season_year = sys.argv[2]
                    if season_year == 'combined':
                        data_file = self.get_data_file_path('combined-seasons.json')
                    else:
                        season_year = int(season_year)
                        data_file = self.get_data_file_path(f'{season_year}-season.json')
                except ValueError:
                    data_file = self.get_data_file_path('2024-season.json')
            else:
                data_file = self.get_data_file_path('2024-season.json')
        
        print("Loading training data...")
        print(f"Loading from: {data_file}")
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)
        
        # Prepare features and labels
        self.X = []
        self.y = []
        self.game_keys = []
        
        for game_key, game_data in self.raw_data.items():
            if 'result' in game_data:
                # Get team keys (exclude result and odds)
                team_keys = [key for key in game_data.keys() 
                           if key not in ['result', 'home_odds', 'away_odds']]
                
                if len(team_keys) == 2:
                    # Extract team stats
                    away_team = team_keys[0]
                    home_team = team_keys[1]
                    
                    away_stats = game_data[away_team]
                    home_stats = game_data[home_team]
                    
                    # Create features
                    features, feature_names = create_comparison_features(away_stats, home_stats)
                    if features is not None:
                        self.X.append(features)
                        self.y.append(game_data['result'])
                        self.game_keys.append(game_key)
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.feature_names = feature_names
        
        print(f"Loaded {len(self.X)} games with {len(self.feature_names)} features")
        return self.X, self.y
    
    def initialize_models(self):
        """
        Initialize all models in the ensemble with optimized hyperparameters.
        """
        print("Initializing ensemble models...")
        
        # XGBoost - often best for sports prediction with optimized parameters
        self.models['xgboost'] = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=1200,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Second XGBoost with different parameters for diversity
        self.models['xgboost_fast'] = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Random Forest - good for interpretability with optimized parameters
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees for diversity
        self.models['extra_trees'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,  # Extra Trees don't use bootstrap
            random_state=42,
            n_jobs=-1
        )
        
        # Logistic Regression - good baseline with regularization
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=0.1,
            penalty='l2',
            solver='liblinear'
        )
        
        # Ridge Classifier for diversity
        self.models['ridge'] = RidgeClassifier(
            random_state=42,
            alpha=0.5
        )
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = cb.CatBoostClassifier(
                iterations=800,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                bootstrap_type='Bernoulli',
                subsample=0.8,
                random_state=42,
                verbose=False
            )
            print("CatBoost model added")
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            )
            print("LightGBM model added")
        
        # Neural Network (load existing trained model)
        try:
            with open('../data/weights.json', 'r') as f:
                weights_data = json.load(f)
            self.models['neural_net'] = PredictionNeuralNetwork(weights_data)
            print("Loaded existing neural network")
        except:
            print("Warning: Could not load neural network weights")
            self.models['neural_net'] = None
        
        print(f"Initialized {len(self.models)} models")
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models in the ensemble.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            X_val (np.array, optional): Validation features for early stopping
            y_val (np.array, optional): Validation labels for early stopping
        """
        print("Training ensemble models...")
        
        # Train each model
        for name, model in self.models.items():
            if model is not None and name != 'neural_net':
                print(f"Training {name}...")
                if name == 'xgboost' and X_val is not None:
                    # Configure XGBoost with early stopping when validation data is available
                    model.set_params(early_stopping_rounds=50)
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)],
                             verbose=False)
                else:
                    model.fit(X_train, y_train)
        
        # Calculate model weights based on cross-validation
        self.calculate_weights(X_train, y_train)
        self.is_trained = True
        print("All models trained successfully!")
    
    def calculate_weights(self, X, y):
        """
        Calculate optimal weights for ensemble based on cross-validation performance.
        
        Args:
            X (np.array): Training features
            y (np.array): Training labels
        """
        print("Calculating ensemble weights...")
        
        cv_scores = {}
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            if model is not None and name != 'neural_net':
                # For XGBoost, temporarily remove early stopping for CV
                if name == 'xgboost':
                    # Create a copy without early stopping for CV
                    cv_model = xgb.XGBClassifier(
                        objective='binary:logistic',
                        n_estimators=100,  # Use fewer trees for CV
                        learning_rate=0.01,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        eval_metric='logloss'
                    )
                else:
                    cv_model = model
                
                scores = cross_val_score(cv_model, X, y, cv=kfold, scoring='accuracy')
                cv_scores[name] = np.mean(scores)
                print(f"{name} CV accuracy: {cv_scores[name]:.4f}")
        
        # Calculate weights based on performance
        total_score = sum(cv_scores.values())
        self.weights = {name: score/total_score for name, score in cv_scores.items()}
        
        # Add neural network weight if available
        if self.models['neural_net'] is not None:
            # Use a default weight for neural network
            self.weights['neural_net'] = 0.2
            # Renormalize weights
            total_weight = sum(self.weights.values())
            self.weights = {name: weight/total_weight for name, weight in self.weights.items()}
        
        print("Ensemble weights:", self.weights)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Args:
            X (np.array): Input features
        
        Returns:
            np.array: Class probabilities of shape (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        all_probs = []
        for name, model in self.models.items():
            if model is not None and name != 'neural_net':
                try:
                    prob = model.predict_proba(X)[:, 1]
                except AttributeError:
                    try:
                        decision_scores = model.decision_function(X)
                        prob = 1 / (1 + np.exp(-decision_scores))
                    except AttributeError:
                        prob = model.predict(X).astype(float)
                all_probs.append(prob)
        
        if all_probs:
            # Weighted average of probabilities
            weights = np.array([self.weights.get(name, 1.0/len(all_probs)) 
                              for name in self.models.keys() 
                              if name != 'neural_net' and self.models[name] is not None])
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_prob = np.zeros(len(X))
            for i, prob in enumerate(all_probs):
                ensemble_prob += weights[i] * prob
            
            return np.column_stack([1 - ensemble_prob, ensemble_prob])
        
        # Final fallback
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
    
    def fit(self, X, y):
        """
        Fit the ensemble model (required for sklearn compatibility).
        
        Args:
            X (np.array): Training features
            y (np.array): Training labels
        """
        # If models are already trained and loaded, don't retrain
        if self.is_trained and len(self.models) > 0:
            # Check if models are actually fitted
            fitted_models = 0
            for name, model in self.models.items():
                if model is not None and hasattr(model, 'classes_'):
                    fitted_models += 1
            
            if fitted_models > 0:
                print(f"Using {fitted_models} pre-trained models (skipping retraining)")
                return self
        
        # Only train if not already trained
        if not self.is_trained:
            self.initialize_models()
            self.train_models(X, y)
        return self
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using ensemble.
        
        Args:
            X (np.array): Input features
            threshold (float): Classification threshold
        
        Returns:
            np.array: Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > threshold).astype(int)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator (required for sklearn compatibility).
        Only include true constructor parameters.
        """
        return {}
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator (required for sklearn compatibility).
        Only include true constructor parameters.
        """
        return self
    
    def evaluate_ensemble(self, X_test, y_test):
        """
        Evaluate ensemble performance on test data.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
        """
        print("Evaluating ensemble performance...")
        
        # Get ensemble predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict_game(self, away_team, home_team):
        """
        Predict the outcome of a specific game.
        
        Args:
            away_team (str): Away team abbreviation
            home_team (str): Home team abbreviation
        
        Returns:
            dict: Prediction results including probabilities and winner
        """
        # Get team stats
        away_stats = get_team_stats(away_team)
        home_stats = get_team_stats(home_team)
        
        # Create features
        features, _ = create_comparison_features(away_stats, home_stats)
        features = features.reshape(1, -1)
        
        # Get ensemble prediction
        probability = self.predict_proba(features)[0, 1]
        prediction = self.predict(features)[0]
        
        # Determine winner
        if prediction == 1:
            winner = away_team
            winner_type = "Away"
        else:
            winner = home_team
            winner_type = "Home"
        
        return {
            'away_team': away_team,
            'home_team': home_team,
            'away_win_probability': probability,
            'home_win_probability': 1 - probability,
            'predicted_winner': winner,
            'winner_type': winner_type,
            'prediction': prediction
        }
    
    def save_ensemble(self, filename='data/ensemble_weights.json'):
        """
        Save ensemble weights and model information to file.
        
        Args:
            filename (str): Path to save the ensemble data
        """
        ensemble_data = {
            'weights': self.weights,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'selected_feature_indices': getattr(self, 'selected_feature_indices', None)
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if ensemble_data['selected_feature_indices'] is not None:
            ensemble_data['selected_feature_indices'] = ensemble_data['selected_feature_indices'].tolist()
        
        with open(filename, 'w') as f:
            json.dump(ensemble_data, f, indent=2)
        print(f"Ensemble saved to {filename}")
    
    def load_ensemble(self, filename='data/ensemble_weights.json'):
        """
        Load ensemble weights and model information from file.
        
        Args:
            filename (str): Path to load the ensemble data from
        """
        try:
            with open(filename, 'r') as f:
                ensemble_data = json.load(f)
            
            self.weights = ensemble_data['weights']
            self.feature_names = ensemble_data['feature_names']
            self.is_trained = ensemble_data['is_trained']
            
            # Convert selected_feature_indices from list back to numpy array
            selected_indices = ensemble_data.get('selected_feature_indices', None)
            if selected_indices is not None:
                self.selected_feature_indices = np.array(selected_indices)
            else:
                self.selected_feature_indices = None
            
            print(f"Ensemble loaded from {filename}")
        except FileNotFoundError:
            print(f"Ensemble file {filename} not found")


def main():
    """
    Main function to demonstrate ensemble model usage.
    """
    print("=== NBA Ensemble Model Demo ===")
    
    # Initialize ensemble
    ensemble = EnsembleNBAPredictor()
    
    # Load data
    X, y = ensemble.load_data()
    
    # Initialize models
    ensemble.initialize_models()
    
    # Split data for training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    ensemble.train_models(X_train, y_train, X_test, y_test)
    
    # Evaluate performance
    accuracy = ensemble.evaluate_ensemble(X_test, y_test)
    
    # Save ensemble
    ensemble.save_ensemble()
    
    print(f"\nEnsemble model training completed with {accuracy:.1%} accuracy!")


if __name__ == "__main__":
    main() 