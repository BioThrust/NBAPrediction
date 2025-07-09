"""
Test Prediction Script

This script loads a season file (e.g., 2025-season.json) and tests the model's predictions
against actual game results for every game in the season.
"""

import json
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Add necessary paths to system path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import custom modules
from utils.shared_utils import get_team_stats, create_comparison_features
from models.ensemble_model import EnsembleNBAPredictor
from models.advanced_ensemble import AdvancedEnsembleNBAPredictor

def get_team_abbreviation(full_name):
    """
    Convert full team name to abbreviation.
    
    Args:
        full_name (str): Full team name (e.g., "Boston Celtics")
    
    Returns:
        str: Team abbreviation (e.g., "BOS")
    """
    team_name_dict = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BRK",
        "Charlotte Hornets": "CHO",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS"
    }
    return team_name_dict.get(full_name, full_name)

def load_model(model_type='advanced', season_year=2025):
    """
    Load the selected prediction model.
    
    Args:
        model_type (str): Type of model to load ('basic', 'advanced')
        season_year (int): Season year for data
    
    Returns:
        model: Loaded prediction model or None if loading fails
    """
    if model_type == 'basic':
        print("Loading basic ensemble model...")
        
        # Try to load saved ensemble weights
        ensemble_weights_found = False
        weights_data = None
        
        # Try year-specific weights first, then generic
        for weights_file in [f'data/{season_year}_ensemble_basic_weights.json', 'data/ensemble_basic_weights.json']:
            try:
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
                accuracy = weights_data.get('model_performance', {}).get('mean_accuracy', 0.0)
                print(f"Found saved ensemble weights! (Accuracy: {accuracy:.1f}%)")
                ensemble_weights_found = True
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading {weights_file}: {e}")
                continue
        
        if not ensemble_weights_found:
            print("No saved ensemble weights found.")
            return None
        
        if ensemble_weights_found and weights_data and ('weights' in weights_data or 'ensemble_weights' in weights_data):
            # Create ensemble model and load weights
            model = EnsembleNBAPredictor()
            model.weights = weights_data.get('weights', weights_data.get('ensemble_weights', {}))
            model.feature_names = weights_data.get('feature_names', [])
            model.is_trained = weights_data.get('is_trained', True)
            model.model_type = 'ensemble'
            
            # Load trained models if available
            models_filename = weights_data.get('models_filename', weights_file.replace('.json', '_models.joblib'))
            if models_filename.startswith('../'):
                models_filename = models_filename[3:]  # Remove '../' prefix
            try:
                trained_models = joblib.load(models_filename)
                model.models.update(trained_models)
                print(f"Loaded {len(trained_models)} trained models: {list(trained_models.keys())}")
            except Exception as e:
                print(f"Warning: Could not load trained models: {e}")
            
            print("Basic ensemble loaded successfully from saved weights!")
            return model
        
    elif model_type == 'advanced':
        print("Loading advanced ensemble model...")
        
        # Try to load saved advanced ensemble weights
        ensemble_weights_found = False
        weights_data = None
        
        # Try year-specific weights first, then generic
        for weights_file in [f'data/{season_year}_ensemble_advanced_weights.json', 'data/ensemble_advanced_weights.json']:
            try:
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
                accuracy = weights_data.get('model_performance', {}).get('mean_accuracy', 0.0)
                print(f"Found saved advanced ensemble weights! (Accuracy: {accuracy:.1f}%)")
                ensemble_weights_found = True
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error loading {weights_file}: {e}")
                continue
        
        if not ensemble_weights_found:
            print("No saved advanced ensemble weights found.")
            return None
        
        if ensemble_weights_found and weights_data and ('weights' in weights_data or 'ensemble_weights' in weights_data):
            # Create advanced ensemble model and load weights
            model = AdvancedEnsembleNBAPredictor()
            model.weights = weights_data.get('weights', weights_data.get('ensemble_weights', {}))
            model.feature_names = weights_data.get('feature_names', [])
            model.is_trained = weights_data.get('is_trained', True)
            if 'betting_thresholds' in weights_data:
                model.betting_thresholds = weights_data['betting_thresholds']
            model.model_type = 'advanced'
            
            # Load selected_feature_indices
            selected_indices = weights_data.get('selected_feature_indices', None)
            if selected_indices is not None:
                model.selected_feature_indices = np.array(selected_indices)
                print(f"Loaded feature selection indices: {len(model.selected_feature_indices)} features selected")
            else:
                model.selected_feature_indices = None
            
            # Load trained models and scaler if available
            models_filename = weights_data.get('models_filename', weights_file.replace('.json', '_models.joblib'))
            if models_filename.startswith('../'):
                models_filename = models_filename[3:]  # Remove '../' prefix
            try:
                save_data = joblib.load(models_filename)
                if isinstance(save_data, dict):
                    # New format with models and scaler
                    trained_models = save_data.get('models', {})
                    model.scaler = save_data.get('scaler', None)
                else:
                    # Old format with just models
                    trained_models = save_data
                    model.scaler = None
                
                model.models.update(trained_models)
                print(f"Loaded {len(trained_models)} trained models: {list(trained_models.keys())}")
                if model.scaler is not None:
                    print("Loaded fitted scaler")
            except Exception as e:
                print(f"Warning: Could not load trained models: {e}")
            
            print("Advanced ensemble loaded successfully from saved weights!")
            return model
    
    return None

def load_season_data(season_file):
    """
    Load season data from JSON file.
    
    Args:
        season_file (str): Path to the season JSON file
    
    Returns:
        dict: Dictionary of game data with game keys
    """
    try:
        with open(season_file, 'r') as f:
            season_data = json.load(f)
        
        print(f"Loaded {len(season_data)} games from {season_file}")
        return season_data
    except FileNotFoundError:
        print(f"Error: Season file {season_file} not found")
        return None
    except Exception as e:
        print(f"Error loading season data: {e}")
        return None

def test_predictions(season_file, model_type='advanced', season_year=2025):
    """
    Test model predictions against actual game results.
    
    Args:
        season_file (str): Path to the season JSON file
        model_type (str): Type of model to use ('basic', 'advanced')
        season_year (int): Season year for data
    """
    print(f"Testing {model_type.upper()} model predictions against {season_file}")
    print("=" * 80)
    
    # Load model
    model = load_model(model_type, season_year)
    if model is None:
        print("❌ Failed to load model. Please ensure the model is trained first.")
        return
    
    # Load season data
    season_data = load_season_data(season_file)
    if season_data is None:
        return
    
    # Initialize counters
    total_games = 0
    correct_predictions = 0
    predictions_data = []
    
    print(f"\n{'Date':<12} {'Away':<4} {'Home':<4} {'Actual':<8} {'Predicted':<10} {'Prob':<6} {'Result':<8}")
    print("-" * 70)
    
    # Process each game
    for game_key, game_data in season_data.items():
        try:
            # Extract game data from the key (format: "TEAM1_vs_TEAM2_DATE")
            parts = game_key.split('_vs_')
            if len(parts) != 2:
                print(f"Warning: Invalid game key format: {game_key}")
                continue
            
            away_abbr = parts[0]
            home_date_part = parts[1]
            
            # Extract home team and date from the second part
            # Format is "HOME_DATE" so we need to find the last underscore
            last_underscore = home_date_part.rfind('_')
            if last_underscore == -1:
                print(f"Warning: Invalid game key format: {game_key}")
                continue
            
            home_abbr = home_date_part[:last_underscore]
            date = home_date_part[last_underscore + 1:]
            
            # Get the actual result from the game data
            actual_result = game_data.get('result')
            if actual_result is None:
                print(f"Warning: No result found for game {game_key}")
                continue
            
            # Get team stats from the game data
            away_stats = game_data.get(away_abbr)
            home_stats = game_data.get(home_abbr)
            
            if away_stats is None or home_stats is None:
                print(f"Warning: Could not load stats for {away_abbr} or {home_abbr}")
                continue
            
            # Create features and make prediction
            features, feature_names = create_comparison_features(away_stats, home_stats)
            
            # Use the selected model for prediction
            if hasattr(model, 'model_type') and model.model_type == 'ensemble':
                # For basic ensemble, we need to process features manually
                features_processed = features.reshape(1, -1)
                
                # Scale features if scaler is available and fitted (for basic ensemble)
                if hasattr(model, 'scaler') and model.scaler is not None:
                    try:
                        features_processed = model.scaler.transform(features_processed)
                    except Exception as e:
                        print(f"Warning: Scaler not fitted properly, using unscaled features: {e}")
                
                # Apply feature selection if the model was trained with feature selection
                if hasattr(model, 'selected_feature_indices') and model.selected_feature_indices is not None:
                    features_processed = features_processed[:, model.selected_feature_indices]
                
                pred = model.predict(features_processed)[0]
                proba = model.predict_proba(features_processed)
                probability = proba[0][1] if len(proba[0]) > 1 else proba[0][0]
                
            elif hasattr(model, 'model_type') and model.model_type == 'advanced':
                # For advanced ensemble, let the model handle feature processing
                try:
                    # Use predict_game_advanced which handles all feature processing internally
                    result = model.predict_game_advanced(away_abbr, home_abbr)
                    pred = result['ensemble_prediction']
                    probability = result['ensemble_probability']
                except (AttributeError, KeyError) as e:
                    print(f"Error with advanced prediction: {e}")
                    continue
            else:
                probability = model.predict_probability(features.reshape(1, -1))[0][0]
                pred = int(probability > 0.5)
            
            # Update counters
            correct = (pred == actual_result)
            total_games += 1
            if correct:
                correct_predictions += 1
            
            # Store prediction data
            prediction_data = {
                'date': date,
                'away': away_abbr,
                'home': home_abbr,
                'actual': actual_result,
                'predicted': pred,
                'probability': probability,
                'correct': correct
            }
            predictions_data.append(prediction_data)
            
            # Print result for current game
            actual_str = "Away" if actual_result == 1 else "Home"
            predicted_str = "Away" if pred == 1 else "Home"
            correct_str = "✓" if correct else "✗"
            
            print(f"{date:<12} {away_abbr:<4} {home_abbr:<4} {actual_str:<8} {predicted_str:<10} {probability:<6.2f} {correct_str:<8}")
            
        except Exception as e:
            print(f"Error processing game {date}: {e}")
            continue
    
    # Calculate and display final results
    accuracy = (correct_predictions / total_games) * 100 if total_games > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"=== FINAL RESULTS ===")
    print(f"Total Games Processed: {total_games}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {total_games - correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Additional analysis by prediction confidence
    if predictions_data:
        # Analyze by prediction confidence levels
        high_confidence = [p for p in predictions_data if p['probability'] > 0.7 or p['probability'] < 0.3]
        medium_confidence = [p for p in predictions_data if 0.4 <= p['probability'] <= 0.6]
        low_confidence = [p for p in predictions_data if (0.3 < p['probability'] < 0.4) or (0.6 < p['probability'] < 0.7)]
        
        print(f"\n=== CONFIDENCE ANALYSIS ===")
        if high_confidence:
            high_acc = sum(1 for p in high_confidence if p['correct']) / len(high_confidence) * 100
            print(f"High Confidence Predictions: {len(high_confidence)} games, {high_acc:.2f}% accuracy")
        
        if medium_confidence:
            med_acc = sum(1 for p in medium_confidence if p['correct']) / len(medium_confidence) * 100
            print(f"Medium Confidence Predictions: {len(medium_confidence)} games, {med_acc:.2f}% accuracy")
        
        if low_confidence:
            low_acc = sum(1 for p in low_confidence if p['correct']) / len(low_confidence) * 100
            print(f"Low Confidence Predictions: {len(low_confidence)} games, {low_acc:.2f}% accuracy")
    
    # Display model information
    print(f"\nModel Type: {model.model_type.upper()}")
    if hasattr(model, 'weights_data') and model.weights_data:
        print(f"Model Training Accuracy: {model.weights_data['model_performance']['mean_accuracy']:.1f}%")
    else:
        print("Model Training Accuracy: N/A (ensemble model)")
    print(f"Actual Performance: {accuracy:.2f}%")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_prediction_results_{model_type}_{timestamp}.json"
    
    results = {
        'model_type': model_type,
        'season_file': season_file,
        'total_games': total_games,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'predictions': predictions_data
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

def main():
    """Main function to handle command line arguments and run the test."""
    if len(sys.argv) < 2:
        print("Usage: python test_prediction.py <season_file> [model_type] [season_year]")
        print("Example: python test_prediction.py 2025-season.json advanced 2025")
        print("Model types: basic, advanced")
        return
    
    season_file = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'advanced'
    season_year = int(sys.argv[3]) if len(sys.argv) > 3 else 2025
    
    if model_type not in ['basic', 'advanced']:
        print("Error: model_type must be 'basic' or 'advanced'")
        return
    
    if not os.path.exists(season_file):
        print(f"Error: Season file {season_file} not found")
        return
    
    test_predictions(season_file, model_type, season_year)

if __name__ == "__main__":
    main() 