"""
NBA Game Prediction Interface

This script provides an interactive interface for predicting NBA game outcomes using trained models.
It supports three model types: Neural Network, Basic Ensemble, and Advanced Ensemble.
Users can input team abbreviations to get predictions with confidence levels.
"""

import json
import numpy as np
import math

# Import custom modules
from ensemble_models.ensemble_model import EnsembleNBAPredictor
from ensemble_models.advanced_ensemble import AdvancedEnsembleNBAPredictor
from utils.shared_utils import PredictionNeuralNetwork, get_team_stats, create_comparison_features


def load_model(model_type='nn'):
    """
    Load the selected prediction model.
    
    Args:
        model_type (str): Type of model to load ('nn', 'ensemble', or 'advanced')
    
    Returns:
        model: Loaded prediction model or None if loading fails
    """
    if model_type == 'ensemble':
        print("Loading basic ensemble model...")
        model = EnsembleNBAPredictor()
        X, y = model.load_data()
        model.initialize_models()
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.train_models(X_train, y_train, X_test, y_test)
        model.model_type = 'ensemble'
        return model
        
    elif model_type == 'advanced':
        print("Loading advanced ensemble model...")
        model = AdvancedEnsembleNBAPredictor()
        X, y = model.load_data()
        model.initialize_models()
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Prepare odds data for training
        odds_train = [model.odds[i] for i in range(len(X)) if i < len(X_train)]
        odds_test = [model.odds[i] for i in range(len(X)) if i >= len(X_train)]
        
        model.train_stacking_ensemble(X_train, y_train, X_test, y_test)
        model.train_voting_ensemble(X_train, y_train)
        model.optimize_betting_thresholds(X_train, y_train, odds_train)
        model.model_type = 'advanced'
        return model
        
    else:
        # Load neural network model
        try:
            with open('json_files/weights.json', 'r') as f:
                weights_data = json.load(f)
            model = PredictionNeuralNetwork(weights_data)
            model.weights_data = weights_data
            model.model_type = 'nn'
            print(f"Neural network loaded successfully! (Accuracy: {weights_data['model_performance']['mean_accuracy']:.1f}%)")
            return model
        except FileNotFoundError:
            print("Error: weights.json not found. Please train the model first using data_collection/sports_binary.py")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


def main():
    """
    Main prediction interface for NBA game predictions.
    Provides an interactive command-line interface for users to input teams and get predictions.
    """
    print("=== NBA Game Predictor ===")
    print("This program predicts the winner of NBA games using trained models.")
    print("Enter team abbreviations (e.g., BOS, LAL, DEN, GSW, PHO, MIA, NYK, MIL, PHI, CLE)")
    print()
    
    # Model selection with ensemble as default
    print("Choose prediction model:")
    print("1. Neural Network (original) - Basic single model")
    print("2. Basic Ensemble (RECOMMENDED) - Combines multiple algorithms for better accuracy")
    print("3. Advanced Ensemble - Includes betting analysis and confidence intervals")
    print()
    print("RECOMMENDED: Choose option 2 for best accuracy or option 3 for betting insights.")
    print()
    
    model_choice = input("Enter 1, 2, or 3 (press Enter for Basic Ensemble): ").strip()
    
    # Default to basic ensemble if no input or invalid input
    if not model_choice or model_choice not in ['1', '2', '3']:
        print("Using Basic Ensemble (recommended)...")
        model_choice = '2'
    
    # Load the selected model
    if model_choice == '2':
        model = load_model('ensemble')
    elif model_choice == '3':
        model = load_model('advanced')
    else:
        model = load_model('nn')
    
    if model is None:
        return
    
    # Check if team stats cache exists
    try:
        with open('json_files/team_stats_cache.json', 'r') as f:
            team_stats_cache = json.load(f)
        available_teams = list(team_stats_cache.keys())
        print(f"Team stats cache loaded successfully! ({len(available_teams)} teams available)")
        print(f"Available teams: {', '.join(available_teams)}")
    except FileNotFoundError:
        print("Warning: team_stats_cache.json not found. Please run data_collection/playoff_data.py first to generate team stats.")
        print("Using placeholder stats for all teams.")
    except Exception as e:
        print(f"Error loading team stats cache: {e}")
        print("Using placeholder stats for all teams.")
    
    print()
    
    # Main prediction loop
    while True:
        try:
            # Get user input for teams
            away_team = input("Enter away team abbreviation: ").upper().strip()
            home_team = input("Enter home team abbreviation: ").upper().strip()
            
            # Validate input
            if away_team == home_team:
                print("Error: Away team and home team cannot be the same!")
                continue
            
            # Get team stats for prediction
            away_stats = get_team_stats(away_team)
            home_stats = get_team_stats(home_team)
            
            # Create features for model input
            features, feature_names = create_comparison_features(away_stats, home_stats)
            
            # Make prediction based on model type
            if hasattr(model, 'model_type') and model.model_type == 'ensemble':
                pred = model.predict(features.reshape(1, -1))[0]
                probability = model.predict_proba(features.reshape(1, -1))[0]
            elif hasattr(model, 'model_type') and model.model_type == 'advanced':
                result = model.predict_with_confidence(features.reshape(1, -1))
                pred = result['ensemble_prediction'][0]
                probability = result['ensemble_probability'][0]
                confidence = result['confidence'][0]
                model_agreement = result['model_agreement'][0]
            else:
                probability = model.predict_probability(features.reshape(1, -1))[0][0]
                pred = int(probability > 0.5)
            
            # Display prediction results
            print(f"\n=== Prediction Results ===")
            print(f"{away_team} @ {home_team}")
            print(f"Home win probability: {(1-probability):.1%}")
            print(f"Away win probability: {probability:.1%}")
            
            # Show predicted winner
            if pred == 0:
                print(f"Predicted winner: {home_team} (Home)")
            else:
                print(f"Predicted winner: {away_team} (Away)")
            
            # Determine confidence level based on probability
            if probability > 0.7 or probability < 0.3:
                confidence_level = "High"
            elif probability > 0.6 or probability < 0.4:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            
            print(f"Confidence: {confidence_level}")
            
            # Show additional info for advanced ensemble
            if hasattr(model, 'model_type') and model.model_type == 'advanced':
                print(f"Model Agreement: {model_agreement:.3f}")
                print(f"Prediction Confidence: {1-confidence:.3f}")
            
            # Show model performance information
            if hasattr(model, 'weights_data') and model.weights_data:
                print(f"Model performance: {model.weights_data['model_performance']['mean_accuracy']:.1f}% accuracy")
            else:
                print("Model performance: Ensemble model (no single accuracy metric)")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with valid team abbreviations.")
        
        # Ask if user wants to predict another game
        print("\n" + "="*50)
        continue_prediction = input("Predict another game? (y/n): ").lower().strip()
        if continue_prediction != 'y':
            print("Thanks for using NBA Game Predictor!")
            break


if __name__ == "__main__":
    main() 