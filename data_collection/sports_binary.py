"""
NBA Neural Network Training Script

This script trains a neural network model for NBA game prediction using team statistics.
It includes feature normalization, comparison feature creation, and cross-validation.
The trained model weights are saved to a JSON file for later use.
"""

import json
import random
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# Load the JSON file into a dictionary
season_data = {}
with open('json_files/2024-season.json', 'r') as f:
    season_data = json.load(f)

def normalize_features(train_data):
    """
    Normalize all features to 0-1 range across the entire dataset.
    
    Args:
        train_data (dict): Dictionary containing game data with team statistics
    
    Returns:
        dict: Normalized data with all features scaled to 0-1 range
    """
    # First pass: find min and max for each feature
    feature_mins = {}
    feature_maxs = {}
    
    # Initialize with first game's values
    first_game_key = list(train_data.keys())[0]
    first_game = train_data[first_game_key]
    team_keys = [key for key in first_game.keys() if key not in ['result', 'home_odds', 'away_odds']]
    
    for team_key in team_keys:
        for feature, value in first_game[team_key].items():
            try:
                float_val = float(value)
                feature_mins[feature] = float_val
                feature_maxs[feature] = float_val
            except:
                pass
    
    # Find global min and max for each feature
    for game_key, game in train_data.items():
        for team_key in [key for key in game.keys() if key not in ['result', 'home_odds', 'away_odds']]:
            for feature, value in game[team_key].items():
                try:
                    float_val = float(value)
                    if feature in feature_mins:
                        feature_mins[feature] = min(feature_mins[feature], float_val)
                        feature_maxs[feature] = max(feature_maxs[feature], float_val)
                except:
                    pass
    
    print("Feature ranges:")
    for feature in feature_mins.keys():
        print(f"  {feature}: {feature_mins[feature]:.2f} to {feature_maxs[feature]:.2f}")
    
    # Second pass: normalize all values
    normalized_data = {}
    for game_key, game in train_data.items():
        normalized_data[game_key] = {}
        
        # Copy result if it exists
        if 'result' in game:
            normalized_data[game_key]['result'] = game['result']
        
        # Normalize team features
        for team_key in [key for key in game.keys() if key not in ['result', 'home_odds', 'away_odds']]:
            normalized_data[game_key][team_key] = {}
            for feature, value in game[team_key].items():
                try:
                    float_val = float(value)
                    if feature_maxs[feature] != feature_mins[feature]:
                        normalized_val = (float_val - feature_mins[feature]) / (feature_maxs[feature] - feature_mins[feature])
                    else:
                        normalized_val = 0.5  # If all values are the same, set to 0.5
                    normalized_data[game_key][team_key][feature] = normalized_val
                except:
                    normalized_data[game_key][team_key][feature] = value
    
    return normalized_data

def create_comparison_features(game):
    """
    Create enhanced features that compare the two teams with sophisticated metrics.
    
    Args:
        game (dict): Game data containing team statistics
    
    Returns:
        tuple: (comparison_features, feature_names) - Feature array and corresponding names
    """
    team_keys = [key for key in game.keys() if key not in ['result', 'home_odds', 'away_odds']]
    if len(team_keys) != 2:
        return None, None
    
    away_team = game[team_keys[0]]  # Away team
    home_team = game[team_keys[1]]  # Home team
    
    comparison_features = []
    feature_names = []
    
    # Basic team comparison features (away - home)
    for feature in away_team.keys():
        if feature in home_team:
            try:
                away_val = float(away_team[feature])
                home_val = float(home_team[feature])
                comparison_features.append(away_val - home_val)
                feature_names.append(f"{feature}_diff")
                comparison_features.append(away_val / (home_val + 1e-8))  # Avoid division by zero
                feature_names.append(f"{feature}_ratio")
            except:
                pass
    
    # Add home court advantage feature
    comparison_features.append(1.0)
    feature_names.append("home_court_advantage")
    
    # Enhanced away win specific features
    try:
        # 1. Net Rating Advantage (most important)
        away_net_rating = float(away_team.get('net_rating', 0))
        home_net_rating = float(home_team.get('net_rating', 0))
        net_rating_advantage = away_net_rating - home_net_rating
        comparison_features.append(net_rating_advantage)
        feature_names.append("net_rating_advantage")
        
        # 2. Away team momentum (only positive when away is better)
        away_momentum = max(0, net_rating_advantage)
        comparison_features.append(away_momentum)
        feature_names.append("away_momentum")
        
        # 3. Offensive vs Defensive Matchup
        away_off_rating = float(away_team.get('offensive_rating', 0))
        home_def_rating = float(home_team.get('defensive_rating', 0))
        offensive_advantage = away_off_rating - home_def_rating
        comparison_features.append(offensive_advantage)
        feature_names.append("offensive_advantage")
        
        # 4. Defensive vs Offensive Matchup
        away_def_rating = float(away_team.get('defensive_rating', 0))
        home_off_rating = float(home_team.get('offensive_rating', 0))
        defensive_advantage = home_off_rating - away_def_rating  # Lower is better for away
        comparison_features.append(defensive_advantage)
        feature_names.append("defensive_advantage")
        
        # 5. Shooting Efficiency Advantage
        away_efg = float(away_team.get('efg_pct', 0))
        home_efg = float(home_team.get('efg_pct', 0))
        shooting_advantage = away_efg - home_efg
        comparison_features.append(shooting_advantage)
        feature_names.append("shooting_advantage")
        
        # 6. Pace Advantage (faster pace might favor away team)
        away_pace = float(away_team.get('pace', 0))
        home_pace = float(home_team.get('pace', 0))
        pace_advantage = away_pace - home_pace
        comparison_features.append(pace_advantage)
        feature_names.append("pace_advantage")
        
        # 7. Turnover Advantage (fewer turnovers is better)
        away_tov = float(away_team.get('offensive_tov', 0))
        home_tov = float(home_team.get('offensive_tov', 0))
        tov_advantage = home_tov - away_tov  # Lower away TOV is better
        comparison_features.append(tov_advantage)
        feature_names.append("turnover_advantage")
        
        # 8. Rebounding Advantage
        away_trb = float(away_team.get('trb', 0))
        home_trb = float(home_team.get('trb', 0))
        rebounding_advantage = away_trb - home_trb
        comparison_features.append(rebounding_advantage)
        feature_names.append("rebounding_advantage")
        
        # 9. Assists Advantage (ball movement)
        away_ast = float(away_team.get('ast', 0))
        home_ast = float(home_team.get('ast', 0))
        assists_advantage = away_ast - home_ast
        comparison_features.append(assists_advantage)
        feature_names.append("assists_advantage")
        
        # 10. Opponent Turnover Forcing Advantage
        away_opp_tov = float(away_team.get('opp_offensive_tov', 0))
        home_opp_tov = float(home_team.get('opp_offensive_tov', 0))
        opp_tov_advantage = away_opp_tov - home_opp_tov  # Higher is better (force more TOVs)
        comparison_features.append(opp_tov_advantage)
        feature_names.append("opp_turnover_advantage")
        
        # 11. Opponent Rebounding Defense Advantage
        away_opp_trb = float(away_team.get('opp_trb', 0))
        home_opp_trb = float(home_team.get('opp_trb', 0))
        opp_reb_advantage = home_opp_trb - away_opp_trb  # Lower opponent rebounds is better
        comparison_features.append(opp_reb_advantage)
        feature_names.append("opp_rebounding_advantage")
        
        # 12. Opponent Assists Defense Advantage
        away_opp_ast = float(away_team.get('opp_ast', 0))
        home_opp_ast = float(home_team.get('opp_ast', 0))
        opp_ast_advantage = home_opp_ast - away_opp_ast  # Lower opponent assists is better
        comparison_features.append(opp_ast_advantage)
        feature_names.append("opp_assists_advantage")
        
        # 13. Opponent Shooting Defense Advantage
        away_opp_efg = float(away_team.get('opp_efg_pct', 0))
        home_opp_efg = float(home_team.get('opp_efg_pct', 0))
        opp_shooting_advantage = home_opp_efg - away_opp_efg  # Lower opponent eFG% is better
        comparison_features.append(opp_shooting_advantage)
        feature_names.append("opp_shooting_advantage")
        
        # 14. Overall Efficiency Rating (weighted combination)
        efficiency_rating = (offensive_advantage * 0.4) + (defensive_advantage * -0.3) + (shooting_advantage * 0.3)
        comparison_features.append(efficiency_rating)
        feature_names.append("overall_efficiency_rating")
        
    except Exception as e:
        print(f"Error creating comparison features: {e}")
        return None, None
    
    return comparison_features, feature_names

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid activation function."""
    return x * (1 - x)

class NeuralNetwork:
    """
    Neural Network class for NBA game prediction.
    
    This class implements a feedforward neural network with one hidden layer
    for binary classification of NBA game outcomes.
    """
    
    def __init__(self, input_size, hidden_size=16, output_size=1):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output neurons (1 for binary classification)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with random values
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        
        # Initialize biases
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X (np.array): Input features
        
        Returns:
            np.array: Network output
        """
        # Hidden layer
        self.layer1 = sigmoid(np.dot(X, self.weights1) + self.bias1)
        
        # Output layer
        self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        
        return self.output
    
    def backward(self, X, y, output):
        """
        Backward pass for gradient descent.
        
        Args:
            X (np.array): Input features
            y (np.array): True labels
            output (np.array): Predicted output
        """
        # Calculate gradients
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.layer1)
        
        # Update weights and biases
        self.weights2 += np.dot(self.layer1.T, output_delta)
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights1 += np.dot(X.T, hidden_delta)
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True)
    
    def backward_weighted(self, X, y, output, weight_dict):
        """
        Backward pass with sample weighting for handling class imbalance.
        
        Args:
            X (np.array): Input features
            y (np.array): True labels
            output (np.array): Predicted output
            weight_dict (dict): Dictionary mapping class labels to weights
        """
        # Apply sample weights
        sample_weights = np.array([weight_dict[label] for label in y.flatten()])
        sample_weights = sample_weights.reshape(-1, 1)
        
        # Calculate weighted gradients
        output_error = (y - output) * sample_weights
        output_delta = output_error * sigmoid_derivative(output)
        
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.layer1)
        
        # Update weights and biases
        self.weights2 += np.dot(self.layer1.T, output_delta)
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights1 += np.dot(X.T, hidden_delta)
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True)
    
    def predict(self, X):
        """Make predictions on new data."""
        return self.forward(X)
    
    def predict_probability(self, X):
        """Get probability predictions on new data."""
        return self.forward(X)

def train_neural_network(X, y, n_epochs=100, verbose=True):
    """
    Train the neural network.
    
    Args:
        X (np.array): Training features
        y (np.array): Training labels
        n_epochs (int): Number of training epochs
        verbose (bool): Whether to print training progress
    
    Returns:
        NeuralNetwork: Trained neural network model
    """
    # Initialize network
    input_size = X.shape[1]
    nn = NeuralNetwork(input_size)
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # Training loop
    for epoch in range(n_epochs):
        # Forward pass
        output = nn.forward(X)
        
        # Backward pass with weighted loss
        nn.backward_weighted(X, y, output, weight_dict)
        
        # Print progress
        if verbose and epoch % 10 == 0:
            predictions = (output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            print(f"Epoch {epoch}, Accuracy: {accuracy:.3f}")
    
    return nn

def cross_validate_neural_network(train_data, n_folds=5, n_epochs=100):
    """
    Perform cross-validation on the neural network.
    
    Args:
        train_data (dict): Training data dictionary
        n_folds (int): Number of cross-validation folds
        n_epochs (int): Number of training epochs per fold
    
    Returns:
        dict: Cross-validation results including accuracies and model weights
    """
    # Prepare data for cross-validation
    X = []
    y = []
    
    for game_key, game in train_data.items():
        if 'result' in game:
            features, feature_names = create_comparison_features(game)
            if features is not None:
                X.append(features)
                y.append(game['result'])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    best_model = None
    best_accuracy = 0
    
    print(f"Starting {n_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = train_neural_network(X_train, y_train, n_epochs=n_epochs, verbose=False)
        
        # Evaluate model
        val_predictions = model.predict(X_val)
        val_predictions_binary = (val_predictions > 0.5).astype(int)
        accuracy = np.mean(val_predictions_binary == y_val)
        fold_accuracies.append(accuracy)
        
        print(f"Fold {fold + 1} Accuracy: {accuracy:.3f}")
        
        # Keep track of best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    # Calculate final statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\nCross-validation results:")
    print(f"Mean Accuracy: {mean_accuracy:.3f} (+/- {std_accuracy:.3f})")
    print(f"Best Accuracy: {best_accuracy:.3f}")
    
    # Save best model weights
    if best_model is not None:
        weights_data = {
            'weights1': best_model.weights1.tolist(),
            'weights2': best_model.weights2.tolist(),
            'bias1': best_model.bias1.tolist(),
            'bias2': best_model.bias2.tolist(),
            'input_size': best_model.input_size,
            'hidden_size': best_model.hidden_size,
            'output_size': best_model.output_size,
            'model_performance': {
                'mean_accuracy': mean_accuracy * 100,
                'std_accuracy': std_accuracy * 100,
                'best_accuracy': best_accuracy * 100,
                'fold_accuracies': [acc * 100 for acc in fold_accuracies]
            }
        }
        
        with open('json_files/weights.json', 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        print("Best model weights saved to json_files/weights.json")
    
    return {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'best_accuracy': best_accuracy,
        'fold_accuracies': fold_accuracies,
        'best_model': best_model
    }

# Main execution
if __name__ == "__main__":
    print("=== NBA Neural Network Training ===")
    
    # Normalize the data
    print("Normalizing features...")
    normalized_data = normalize_features(season_data)
    
    # Perform cross-validation
    cv_results = cross_validate_neural_network(normalized_data, n_folds=5, n_epochs=100)
    
    print("\nTraining completed successfully!")
    print(f"Final model accuracy: {cv_results['mean_accuracy']:.1%}")