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
    
    def __init__(self):
        """
        Initialize ensemble model with multiple algorithms.
        """
        self.models = {}
        self.weights = {}
        self.feature_names = None
        self.is_trained = False
    
    def load_data(self, data_file='json_files/2024-season.json'):
        """
        Load and prepare training data from JSON file.
        
        Args:
            data_file (str): Path to the JSON file containing game data
        
        Returns:
            tuple: (X, y) - Feature matrix and target labels
        """
        print("Loading training data...")
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
        
        # XGBoost - often best for sports prediction
        self.models['xgboost'] = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Random Forest - good for interpretability
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Logistic Regression - good baseline
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        
        # Neural Network (load existing trained model)
        try:
            with open('json_files/weights.json', 'r') as f:
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
        Get weighted probability predictions from all models.
        
        Args:
            X (np.array): Input features
        
        Returns:
            np.array: Weighted ensemble probabilities
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if model is not None:
                if name == 'neural_net':
                    pred = model.predict_probability(X)[:, 0]  # Neural net returns different format
                else:
                    pred = model.predict_proba(X)[:, 1]  # Probability of positive class
                predictions[name] = pred
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
    
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
        return (probabilities > threshold).astype(int)
    
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
        probability = self.predict_proba(features)[0]
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
    
    def save_ensemble(self, filename='json_files/ensemble_weights.json'):
        """
        Save ensemble weights and model information to file.
        
        Args:
            filename (str): Path to save the ensemble data
        """
        ensemble_data = {
            'weights': self.weights,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filename, 'w') as f:
            json.dump(ensemble_data, f, indent=2)
        print(f"Ensemble saved to {filename}")
    
    def load_ensemble(self, filename='json_files/ensemble_weights.json'):
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