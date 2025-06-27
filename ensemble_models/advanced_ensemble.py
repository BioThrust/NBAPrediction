import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.shared_utils import get_team_stats, create_comparison_features, PredictionNeuralNetwork
import warnings
warnings.filterwarnings('ignore')

def get_data_file_path(filename):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'json_files', filename)

class AdvancedEnsembleNBAPredictor:
    def __init__(self):
        """Initialize advanced ensemble with stacking and multiple algorithms"""
        self.models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.betting_thresholds = {}
        
    def load_data(self, data_file=None):
        """Load and prepare training data with odds information"""
        if data_file is None:
            import sys
            if len(sys.argv) > 2:
                try:
                    season_year = sys.argv[2]
                    if season_year == 'combined':
                        data_file = get_data_file_path('combined-seasons.json')
                    else:
                        season_year = int(season_year)
                        data_file = get_data_file_path(f'{season_year}-season.json')
                except ValueError:
                    data_file = get_data_file_path('2024-season.json')
            else:
                data_file = get_data_file_path('2024-season.json')
        
        print("Loading training data...")
        print(f"Loading from: {data_file}")
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)
        
        # Prepare features and labels
        self.X = []
        self.y = []
        self.odds = []
        self.game_keys = []
        feature_names = None  # Initialize feature_names
        
        for game_key, game_data in self.raw_data.items():
            if 'result' in game_data:
                team_keys = [key for key in game_data.keys() 
                           if key not in ['result', 'home_odds', 'away_odds']]
                
                if len(team_keys) == 2:
                    # Extract team stats
                    away_team = team_keys[0]
                    home_team = team_keys[1]
                    
                    away_stats = game_data[away_team]
                    home_stats = game_data[home_team]
                    
                    # Create features
                    features, current_feature_names = create_comparison_features(away_stats, home_stats)
                    if features is not None:
                        # Set feature_names on first iteration
                        if feature_names is None:
                            feature_names = current_feature_names
                        
                        self.X.append(features)
                        self.y.append(game_data['result'])
                        
                        # Store odds if available
                        home_odds = game_data.get('home_odds', None)
                        away_odds = game_data.get('away_odds', None)
                        self.odds.append({'home': home_odds, 'away': away_odds})
                        
                        self.game_keys.append(game_key)
        
        # Check if we have any data
        if len(self.X) == 0:
            raise ValueError("No valid games found in dataset. Please check your data file.")
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.feature_names = feature_names
        
        # Scale features
        self.X = self.scaler.fit_transform(self.X)
        
        print(f"Loaded {len(self.X)} games with {len(self.feature_names)} features")
        return self.X, self.y
    
    def initialize_models(self):
        """Initialize diverse set of models for ensemble"""
        print("Initializing advanced ensemble models...")
        
        # Base models with different characteristics
        self.models = {
            # Gradient Boosting models
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            
            # Tree-based models
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            # Linear models
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            
            'ridge': RidgeClassifier(
                random_state=42,
                alpha=1.0
            ),
            
            # SVM
            'svm': SVC(
                probability=True,
                random_state=42,
                kernel='rbf',
                C=1.0
            )
        }
        
        # Load neural network if available
        try:
            with open('../json_files/weights.json', 'r') as f:
                weights_data = json.load(f)
            self.models['neural_net'] = PredictionNeuralNetwork(weights_data)
            print("Loaded existing neural network")
        except:
            print("Warning: Could not load neural network weights")
        
        print(f"Initialized {len(self.models)} models")
    
    def train_stacking_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Train stacking ensemble with meta-learner"""
        print("Training stacking ensemble...")
        
        # Train base models
        base_predictions = {}
        
        for name, model in self.models.items():
            if model is not None and name != 'neural_net':
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Get predictions for meta-learner
                if X_val is not None:
                    try:
                        # Try predict_proba first
                        base_predictions[name] = model.predict_proba(X_val)[:, 1]
                    except AttributeError:
                        # For models without predict_proba (like RidgeClassifier), use decision_function
                        try:
                            decision_scores = model.decision_function(X_val)
                            # Convert decision scores to probabilities using sigmoid
                            base_predictions[name] = 1 / (1 + np.exp(-decision_scores))
                        except AttributeError:
                            # Fallback to binary predictions
                            preds = model.predict(X_val)
                            base_predictions[name] = preds.astype(float)
                else:
                    # Use cross-validation predictions
                    cv_preds = cross_val_score(model, X_train, y_train, 
                                             cv=5, scoring='roc_auc')
                    print(f"{name} CV ROC-AUC: {np.mean(cv_preds):.4f}")
        
        # Train meta-learner if validation data available
        if X_val is not None and len(base_predictions) > 0:
            meta_features = np.column_stack(list(base_predictions.values()))
            self.meta_model = LogisticRegression(random_state=42)
            self.meta_model.fit(meta_features, y_val)
            print("Meta-learner trained")
        
        self.is_trained = True
        print("Stacking ensemble trained successfully!")
    
    def train_voting_ensemble(self, X_train, y_train):
        """Train voting ensemble with different voting strategies"""
        print("Training voting ensemble...")
        
        # Create voting classifier
        estimators = [(name, model) for name, model in self.models.items() 
                     if model is not None and name != 'neural_net']
        
        self.voting_soft = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=None  # Equal weights initially
        )
        
        self.voting_hard = VotingClassifier(
            estimators=estimators,
            voting='hard'
        )
        
        # Train voting classifiers
        self.voting_soft.fit(X_train, y_train)
        self.voting_hard.fit(X_train, y_train)
        
        print("Voting ensembles trained")
    
    def optimize_betting_thresholds(self, X_train, y_train, odds_data):
        """Optimize betting thresholds using Kelly Criterion"""
        print("Optimizing betting thresholds...")
        
        # Convert odds_data to numpy array for indexing
        odds_data = np.array(odds_data)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            if model is not None and name != 'neural_net':
                try:
                    pred_proba = model.predict_proba(X_train)[:, 1]
                except AttributeError:
                    # For models without predict_proba (like RidgeClassifier), use decision_function
                    try:
                        decision_scores = model.decision_function(X_train)
                        # Convert decision scores to probabilities using sigmoid
                        pred_proba = 1 / (1 + np.exp(-decision_scores))
                    except AttributeError:
                        # Fallback to binary predictions
                        preds = model.predict(X_train)
                        pred_proba = preds.astype(float)
                
                predictions[name] = pred_proba
        
        # Calculate Kelly Criterion for different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        kelly_results = {}
        
        for name, preds in predictions.items():
            kelly_scores = []
            for threshold in thresholds:
                # Get games where we bet
                bet_mask = (preds > threshold) | (preds < (1 - threshold))
                if np.sum(bet_mask) > 0:
                    # Calculate Kelly Criterion
                    kelly_score = self.calculate_kelly_criterion(
                        preds[bet_mask], y_train[bet_mask], 
                        odds_data[bet_mask], threshold
                    )
                    kelly_scores.append(kelly_score)
                else:
                    kelly_scores.append(0)
            
            # Find optimal threshold
            optimal_idx = np.argmax(kelly_scores)
            self.betting_thresholds[name] = {
                'threshold': thresholds[optimal_idx],
                'kelly_score': kelly_scores[optimal_idx]
            }
            print(f"{name} optimal threshold: {thresholds[optimal_idx]:.2f} "
                  f"(Kelly score: {kelly_scores[optimal_idx]:.4f})")
    
    def calculate_kelly_criterion(self, predictions, actuals, odds_data, threshold):
        """Calculate Kelly Criterion for betting strategy"""
        total_bets = 0
        total_profit = 0
        
        for pred, actual, odds in zip(predictions, actuals, odds_data):
            if pred > threshold:  # Bet on home team
                if odds['home'] is not None:
                    bet_amount = 1.0
                    if actual == 1:  # Home team wins
                        profit = bet_amount * (odds['home'] - 1)
                    else:
                        profit = -bet_amount
                    total_bets += bet_amount
                    total_profit += profit
            elif pred < (1 - threshold):  # Bet on away team
                if odds['away'] is not None:
                    bet_amount = 1.0
                    if actual == 0:  # Away team wins
                        profit = bet_amount * (odds['away'] - 1)
                    else:
                        profit = -bet_amount
                    total_bets += bet_amount
                    total_profit += profit
        
        if total_bets > 0:
            return total_profit / total_bets
        return 0
    
    def predict_with_confidence(self, X):
        """Make predictions with confidence intervals"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if model is not None and name != 'neural_net':
                pred = model.predict(X)
                predictions[name] = pred
                
                # Get probabilities
                try:
                    prob = model.predict_proba(X)[:, 1]
                    probabilities[name] = prob
                except AttributeError:
                    # For models without predict_proba (like RidgeClassifier), use decision_function
                    try:
                        decision_scores = model.decision_function(X)
                        # Convert decision scores to probabilities using sigmoid
                        prob = 1 / (1 + np.exp(-decision_scores))
                        probabilities[name] = prob
                    except AttributeError:
                        # Fallback to binary predictions as probabilities
                        prob = pred.astype(float)
                        probabilities[name] = prob
        
        # Calculate ensemble prediction
        if len(predictions) > 0:
            # Weighted average based on model performance
            weights = np.ones(len(predictions)) / len(predictions)
            ensemble_prob = np.zeros(len(X))
            
            for i, (name, prob) in enumerate(probabilities.items()):
                ensemble_prob += weights[i] * prob
            
            # Calculate confidence (standard deviation of predictions)
            prob_array = np.array(list(probabilities.values()))
            confidence = np.std(prob_array, axis=0)
            
            return {
                'ensemble_probability': ensemble_prob,
                'ensemble_prediction': (ensemble_prob > 0.5).astype(int),
                'confidence': confidence,
                'model_agreement': 1 - confidence,  # Higher agreement = lower std
                'individual_predictions': predictions,
                'individual_probabilities': probabilities
            }
        
        return None
    
    def get_betting_recommendations(self, prediction_result, odds_data):
        """Get betting recommendations using Kelly Criterion"""
        recommendations = []
        
        for i, (prob, conf) in enumerate(zip(prediction_result['ensemble_probability'], 
                                           prediction_result['confidence'])):
            # Only bet if confidence is high enough
            if conf < 0.2:  # Low standard deviation = high confidence
                kelly_fraction = self.calculate_kelly_fraction(prob, odds_data[i])
                
                if kelly_fraction > 0.05:  # Only bet if Kelly fraction is significant
                    bet_side = "HOME" if prob > 0.5 else "AWAY"
                    bet_amount = kelly_fraction
                    
                    recommendations.append({
                        'game_index': i,
                        'bet_side': bet_side,
                        'confidence': 1 - conf,
                        'kelly_fraction': kelly_fraction,
                        'recommended_bet': bet_amount,
                        'model_probability': prob
                    })
        
        return recommendations
    
    def calculate_kelly_fraction(self, probability, odds):
        """Calculate Kelly Criterion betting fraction"""
        if probability > 0.5:  # Bet on home team
            if odds['home'] is not None:
                b = odds['home'] - 1  # Net odds
                p = probability
                q = 1 - probability
                kelly = (b * p - q) / b
                return max(0, kelly)  # Don't bet if negative
        else:  # Bet on away team
            if odds['away'] is not None:
                b = odds['away'] - 1  # Net odds
                p = 1 - probability
                q = probability
                kelly = (b * p - q) / b
                return max(0, kelly)  # Don't bet if negative
        
        return 0
    
    def evaluate_ensemble_performance(self, X_test, y_test, odds_test=None):
        """Comprehensive ensemble evaluation"""
        print("Evaluating advanced ensemble performance...")
        
        # Get predictions
        prediction_result = self.predict_with_confidence(X_test)
        
        # Basic accuracy metrics
        ensemble_acc = accuracy_score(y_test, prediction_result['ensemble_prediction'])
        ensemble_auc = roc_auc_score(y_test, prediction_result['ensemble_probability'])
        
        print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
        print(f"Ensemble ROC-AUC: {ensemble_auc:.4f}")
        
        # Individual model performance
        for name, preds in prediction_result['individual_predictions'].items():
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, prediction_result['individual_probabilities'][name])
            print(f"{name}: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}")
        
        # Betting performance if odds available
        if odds_test is not None:
            recommendations = self.get_betting_recommendations(prediction_result, odds_test)
            
            if recommendations:
                total_bets = len(recommendations)
                winning_bets = 0
                total_profit = 0
                total_money_bet = 0
                
                for rec in recommendations:
                    game_idx = rec['game_index']
                    bet_side = rec['bet_side']
                    bet_amount = rec['recommended_bet']
                    total_money_bet += bet_amount
                    
                    if bet_side == "HOME" and y_test[game_idx] == 0:
                        winning_bets += 1
                        if odds_test[game_idx]['home'] is not None:
                            total_profit += bet_amount * (odds_test[game_idx]['home'] - 1)
                    elif bet_side == "AWAY" and y_test[game_idx] == 1:
                        winning_bets += 1
                        if odds_test[game_idx]['away'] is not None:
                            total_profit += bet_amount * (odds_test[game_idx]['away'] - 1)
                    else:
                        total_profit -= bet_amount
                
                bet_accuracy = winning_bets / total_bets if total_bets > 0 else 0
                print(f"\nBetting Performance:")
                print(f"Total bets: {total_bets}")
                print(f"Total money bet: {total_money_bet:.4f}")
                print(f"Betting accuracy: {bet_accuracy:.4f}")
                print(f"Total profit: {total_profit:.4f}")
                print(f"ROI: {(total_profit/total_money_bet)*100:.2f}%" if total_money_bet > 0 else "N/A")
        
        return ensemble_acc, ensemble_auc
    
    def predict_game_advanced(self, away_team, home_team, odds=None):
        """Advanced single game prediction with betting analysis"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Get team stats and create features
        away_stats = get_team_stats(away_team)
        home_stats = get_team_stats(home_team)
        features, _ = create_comparison_features(away_stats, home_stats)
        features = self.scaler.transform(features.reshape(1, -1))
        
        # Get ensemble prediction
        prediction_result = self.predict_with_confidence(features)
        
        # Prepare odds data
        odds_data = [{'home': odds['home'] if odds else None, 
                     'away': odds['away'] if odds else None}]
        
        # Get betting recommendations
        recommendations = self.get_betting_recommendations(prediction_result, odds_data)
        
        return {
            'away_team': away_team,
            'home_team': home_team,
            'ensemble_probability': prediction_result['ensemble_probability'][0],
            'ensemble_prediction': prediction_result['ensemble_prediction'][0],
            'confidence': prediction_result['confidence'][0],
            'model_agreement': prediction_result['model_agreement'][0],
            'individual_predictions': {k: v[0] for k, v in prediction_result['individual_predictions'].items()},
            'individual_probabilities': {k: v[0] for k, v in prediction_result['individual_probabilities'].items()},
            'betting_recommendations': recommendations
        }

def main():
    """Main function to train and evaluate advanced ensemble"""
    print("=== Advanced NBA Ensemble Model Training ===")
    
    # Initialize ensemble
    ensemble = AdvancedEnsembleNBAPredictor()
    
    # Load data
    X, y = ensemble.load_data()
    
    # Initialize models
    ensemble.initialize_models()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split odds data accordingly
    odds_train = [ensemble.odds[i] for i in range(len(X)) if i < len(X_train)]
    odds_test = [ensemble.odds[i] for i in range(len(X)) if i >= len(X_train)]
    
    # Train ensembles
    ensemble.train_stacking_ensemble(X_train, y_train, X_test, y_test)
    ensemble.train_voting_ensemble(X_train, y_train)
    
    # Optimize betting thresholds
    ensemble.optimize_betting_thresholds(X_train, y_train, odds_train)
    
    # Evaluate performance
    acc, auc = ensemble.evaluate_ensemble_performance(X_test, y_test, odds_test)
    
    # Test single game prediction
    print("\n=== Advanced Single Game Prediction Test ===")
    test_odds = {'home': 1.85, 'away': 2.05}  # Example odds
    prediction = ensemble.predict_game_advanced("BOS", "LAL", test_odds)
    
    print(f"Game: {prediction['away_team']} @ {prediction['home_team']}")
    print(f"Ensemble Prediction: {prediction['ensemble_prediction']} "
          f"(Probability: {prediction['ensemble_probability']:.3f})")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"Model Agreement: {prediction['model_agreement']:.3f}")
    
    print("\nIndividual Model Predictions:")
    for model_name, model_pred in prediction['individual_predictions'].items():
        prob = prediction['individual_probabilities'][model_name]
        print(f"  {model_name}: {model_pred} (Probability: {prob:.3f})")
    
    if prediction['betting_recommendations']:
        print("\nBetting Recommendations:")
        for rec in prediction['betting_recommendations']:
            print(f"  Bet on {rec['bet_side']}: {rec['recommended_bet']:.2f} "
                  f"(Confidence: {rec['confidence']:.3f})")
    else:
        print("\nNo betting recommendations (insufficient confidence)")
    
    print(f"\nFinal Ensemble Accuracy: {acc:.4f}")
    print(f"Final Ensemble ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main() 