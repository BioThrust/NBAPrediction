"""
Enhanced NBA Game Prediction Script

This script uses improved ensemble models with enhanced features, feature selection,
and model calibration to predict NBA game outcomes with higher accuracy.
"""

import sys
import os
import json
import numpy as np
import config

# Add necessary paths to system path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import custom modules
from utils.shared_utils import get_team_stats, create_comparison_features, select_best_features
from models.ensemble_model import EnsembleNBAPredictor
from models.advanced_ensemble import AdvancedEnsembleNBAPredictor

def predict_game_enhanced(away_team, home_team, model_type='advanced', season_year=None):
    """
    Enhanced prediction function with improved accuracy.
    
    Args:
        away_team (str): Away team abbreviation (e.g., "BOS")
        home_team (str): Home team abbreviation (e.g., "LAL")
        model_type (str): Type of model to use ('basic', 'advanced')
        season_year (int): Season year for data
    
    Returns:
        dict: Prediction results with confidence and betting analysis
    """
    if season_year is None:
        season_year = config.SEASON_YEAR
    
    print(f"Making enhanced prediction for {away_team} @ {home_team} ({season_year} season)")
    print("=" * 60)
    
    # Get team statistics
    try:
        away_stats = get_team_stats(away_team, season_year)
        home_stats = get_team_stats(home_team, season_year)
        
        if away_stats is None or home_stats is None:
            print(f"Error: Could not load team statistics for {away_team} or {home_team}")
            return None
            
    except Exception as e:
        print(f"Error loading team stats: {e}")
        return None
    
    # Create enhanced features
    features, feature_names = create_comparison_features(away_stats, home_stats)
    
    if features is None:
        print("Error: Could not create features")
        return None
    
    # Load and use the appropriate model
    if model_type == 'advanced':
        print("Using Advanced Ensemble Model...")
        try:
            # Try to load saved advanced ensemble
            model = AdvancedEnsembleNBAPredictor()
            
            # Check for saved weights
            weights_file = f'data/{season_year}_ensemble_advanced_weights.json'
            if os.path.exists(weights_file):
                model.load_ensemble(weights_file)
                if model.is_trained:
                    print("Loaded pre-trained advanced ensemble")
                else:
                    print("Training advanced ensemble from scratch...")
                    X, y = model.load_data()
                    model.initialize_models()
                    
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    model.train_stacking_ensemble(X_train, y_train, X_test, y_test)
                    model.train_voting_ensemble(X_train, y_train)
                    
                    # Save the trained model
                    model.save_ensemble(weights_file)
            else:
                print("Training advanced ensemble from scratch...")
                X, y = model.load_data()
                model.initialize_models()
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                model.train_stacking_ensemble(X_train, y_train, X_test, y_test)
                model.train_voting_ensemble(X_train, y_train)
                
                # Save the trained model
                model.save_ensemble(weights_file)
            
            # Make prediction
            prediction_result = model.predict_game_advanced(away_team, home_team)
            
        except Exception as e:
            print(f"Error with advanced ensemble: {e}")
            print("Falling back to basic ensemble...")
            model_type = 'basic'
    
    if model_type == 'basic':
        print("Using Basic Ensemble Model...")
        try:
            model = EnsembleNBAPredictor()
            
            # Check for saved weights
            weights_file = f'data/{season_year}_ensemble_basic_weights.json'
            if os.path.exists(weights_file):
                model.load_ensemble(weights_file)
                if model.is_trained:
                    print("Loaded pre-trained basic ensemble")
                else:
                    print("Training basic ensemble from scratch...")
                    X, y = model.load_data()
                    model.initialize_models()
                    
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    model.train_models(X_train, y_train, X_test, y_test)
                    model.save_ensemble(weights_file)
            else:
                print("Training basic ensemble from scratch...")
                X, y = model.load_data()
                model.initialize_models()
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                model.train_models(X_train, y_train, X_test, y_test)
                model.save_ensemble(weights_file)
            
            # Make prediction
            prediction_result = model.predict_game(away_team, home_team)
            
        except Exception as e:
            print(f"Error with basic ensemble: {e}")
            return None
    
    # Display enhanced results
    if prediction_result:
        print("\n" + "=" * 60)
        print("ENHANCED PREDICTION RESULTS")
        print("=" * 60)
        
        # Display team comparison
        print(f"\nTeam Comparison ({season_year} Season):")
        print(f"{away_team} (Away) vs {home_team} (Home)")
        print("-" * 40)
        
        # Key metrics comparison
        metrics = ['net_rating', 'offensive_rating', 'defensive_rating', 'efg_pct', 'pace']
        for metric in metrics:
            away_val = away_stats.get(metric, 0)
            home_val = home_stats.get(metric, 0)
            diff = away_val - home_val
            print(f"{metric.replace('_', ' ').title()}: {away_val:.1f} vs {home_val:.1f} (diff: {diff:+.1f})")
        
        # --- Always print prediction result ---
        print("\nPrediction Result:")
        winner = None
        prob = None
        confidence = None
        # Advanced ensemble format
        if 'ensemble_probability' in prediction_result:
            prob = prediction_result['ensemble_probability']
            pred = prediction_result.get('ensemble_prediction', int(prob > 0.5))
            confidence = prediction_result.get('confidence', abs(prob - 0.5) * 2)
            winner = away_team if pred else home_team
        # Basic ensemble format
        elif 'away_win_probability' in prediction_result:
            prob = prediction_result['away_win_probability']
            pred = prediction_result.get('prediction', int(prob > 0.5))
            confidence = abs(prob - 0.5) * 2
            winner = away_team if pred else home_team
        # Fallback
        else:
            prob = prediction_result.get('probability', None)
            if prob is not None:
                pred = int(prob > 0.5)
                confidence = abs(prob - 0.5) * 2
                winner = away_team if pred else home_team
        if prob is not None:
            print(f"Predicted Winner: {winner}")
            print(f"Probability (away win): {prob:.1%}")
            print(f"Confidence: {confidence:.1%}")
        else:
            print("Prediction failed or result format unknown.")
        
        # Additional analysis
        if 'model_details' in prediction_result:
            print(f"\nModel Details:")
            for key, value in prediction_result['model_details'].items():
                print(f"  {key}: {value}")
        
        return prediction_result
    
    return None

def main():
    """Main function to run enhanced predictions"""
    if len(sys.argv) < 3:
        print("Usage: python predict_game.py <away_team> <home_team> [model_type] [season_year]")
        print("Example: python predict_game.py BOS LAL advanced 2025")
        print("\nModel types: basic, advanced")
        print("Season years: 2024, 2025, etc.")
        return
    
    away_team = sys.argv[1].upper()
    home_team = sys.argv[2].upper()
    model_type = sys.argv[3] if len(sys.argv) > 3 else 'advanced'
    season_year = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    # Validate inputs
    valid_teams = list(config.TEAM_ABBR_TO_FULL.keys())
    if away_team not in valid_teams or home_team not in valid_teams:
        print("Error: Invalid team abbreviation")
        print(f"Valid teams: {', '.join(valid_teams)}")
        return
    
    if model_type not in ['basic', 'advanced']:
        print("Error: Model type must be 'basic' or 'advanced'")
        return
    
    # Make prediction
    result = predict_game_enhanced(away_team, home_team, model_type, season_year)
    
    if result:
        print(f"\n✅ Enhanced prediction completed successfully!")
    else:
        print(f"\n❌ Prediction failed. Please check your inputs and try again.")

if __name__ == "__main__":
    main() 