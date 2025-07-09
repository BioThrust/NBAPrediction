"""
NBA Prediction Model Testing and Results Analysis

This script compares the performance of different NBA prediction models against actual game results.
It supports three model types: Neural Network, Basic Ensemble, and Advanced Ensemble.
The script also calculates betting performance and ROI based on prediction confidence.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import config
import joblib

# Add necessary paths to system path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_collection')))

# Import custom modules
from data_collection.basketball_reference_scraper.seasons import get_schedule
from data_collection.basketball_reference_scraper.players import get_game_logs
from utils.shared_utils import PredictionNeuralNetwork, get_team_stats, create_comparison_features
from models.ensemble_model import EnsembleNBAPredictor
from models.advanced_ensemble import AdvancedEnsembleNBAPredictor

# Load playoff dataset with odds information
season_year = 2024  # Default season
if len(sys.argv) > 1:
    try:
        season_year = int(sys.argv[1])
    except ValueError:
        print("Error: Season year must be a valid integer (e.g., 2024)")
        sys.exit(1)

season_data_file = f'data/{season_year}-season.json'


def load_model(model_type='nn', season_year=2025):
    """
    Load the selected prediction model.
    
    Args:
        model_type (str): Type of model to load ('nn', 'ensemble', or 'advanced')
        season_year (int): Season year for data
    
    Returns:
        model: Loaded prediction model or None if loading fails
    """
    if model_type == 'ensemble':
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
            print("No saved ensemble weights found. Training from scratch...")
        
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
        
        # Fallback to training from scratch
        model = EnsembleNBAPredictor()
        X, y = model.load_data()
        model.fit(X, y)  # Use the new fit method
        model.model_type = 'ensemble'
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
            print("No saved advanced ensemble weights found. Training from scratch...")
        
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
        
        # Fallback to training from scratch
        model = AdvancedEnsembleNBAPredictor()
        X, y = model.load_data()
        model.fit(X, y)  # Use the new fit method
        model.model_type = 'advanced'
        return model
        
    else:
        # Load neural network model
        try:
            with open('data/weights.json', 'r') as f:
                weights_data = json.load(f)
            model = PredictionNeuralNetwork(weights_data)
            model.weights_data = weights_data
            model.model_type = 'nn'
            print(f"Model loaded successfully! (Training Accuracy: {weights_data['model_performance']['mean_accuracy']:.1f}%)")
            return model
        except FileNotFoundError:
            print("Error: weights.json not found. Please train the model first using data_collection/sports_binary.py")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


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


def get_team_full_name(abbreviation):
    """
    Convert team abbreviation to full name.
    
    Args:
        abbreviation (str): Team abbreviation (e.g., "BOS")
    
    Returns:
        str: Full team name (e.g., "Boston Celtics")
    """
    team_abbr_dict = {
        "ATL": "Atlanta Hawks",
        "BOS": "Boston Celtics",
        "BRK": "Brooklyn Nets",
        "CHO": "Charlotte Hornets",
        "CHI": "Chicago Bulls",
        "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks",
        "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors",
        "HOU": "Houston Rockets",
        "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers",
        "LAL": "Los Angeles Lakers",
        "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks",
        "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans",
        "NYK": "New York Knicks",
        "OKC": "Oklahoma City Thunder",
        "ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers",
        "PHO": "Phoenix Suns",
        "POR": "Portland Trail Blazers",
        "SAC": "Sacramento Kings",
        "SAS": "San Antonio Spurs",
        "TOR": "Toronto Raptors",
        "UTA": "Utah Jazz",
        "WAS": "Washington Wizards"
    }
    return team_abbr_dict.get(abbreviation.upper(), abbreviation)


# List of star players to monitor for injuries
star_players = [
    "Donovan Mitchell",
    "Jayson Tatum",
    "Jalen Brunson",
    "Tyrese Haliburton",
    "Giannis Antetokounmpo",
    "Cade Cunningham",
    "Jimmy Butler",
    "Shai Gilgeous-Alexander",
    "Luka Doncic",
    "LeBron James",
    "Nikola Jokic",
    "Anthony Edwards",
]


def pre_calculate_test_results(season_year, model_type='advanced'):
    """
    Add actual game results from Basketball Reference to the season JSON file.
    
    Args:
        season_year (int): Season year
        model_type (str): Type of model (not used, kept for compatibility)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Adding actual results to {season_year} season data...")
    
    # Load season data
    season_file = f'data/{season_year}-season.json'
    try:
        with open(season_file, 'r') as f:
            season_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Season file {season_file} not found")
        return False
    
    # Get schedule data for actual results
    print("Loading schedule data for actual results...")
    matchups = get_schedule(season_year)
    matchups = remove_injuries(matchups)
    
    # Create a mapping of game keys to actual results
    actual_results = {}
    for index, row in matchups.iterrows():
        visitor = row['VISITOR']
        home = row['HOME']
        visitor_pts = row['VISITOR_PTS']
        home_pts = row['HOME_PTS']
        date = row['DATE'].strftime('%Y-%m-%d')
        print(f"DEBUG: Visitor: {visitor}, Home: {home}, Visitor Pts: {visitor_pts}, Home Pts: {home_pts}, Date: {date}")
        # Skip games without scores
        if pd.isna(visitor_pts) or pd.isna(home_pts):
            continue
        
        # Determine actual result (1 = away win, 0 = home win)
        if visitor_pts > home_pts:
            actual_result = 1
        else:
            actual_result = 0
        
        # Get team abbreviations
        away_abbr = get_team_abbreviation(visitor)
        home_abbr = get_team_abbreviation(home)
        
        # Create game key
        game_key = f"{away_abbr}_vs_{home_abbr}_{date}"
        
        actual_results[game_key] = {
            'actual_result': actual_result,
            'away_pts': visitor_pts,
            'home_pts': home_pts
        }
    
    print(f"Found {len(actual_results)} games with actual results")
    
    # Add actual results to season data
    games_processed = 0
    for game_key, game_data in season_data.items():
        if game_key in actual_results:
            game_data['test_result'] = actual_results[game_key]
            games_processed += 1
    
    # Save updated season data
    try:
        with open(season_file, 'w') as f:
            json.dump(season_data, f, indent=2)
        print(f"✅ Successfully added actual results to {games_processed} games in {season_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving actual results: {e}")
        return False

def remove_injuries(matchups, star_players=None):
    """
    Remove games where either team has a major injury to star players.
    
    Args:
        matchups (DataFrame): DataFrame containing game matchups
        star_players (list, optional): List of star players to check for injuries. Defaults to config.STAR_PLAYERS.
    
    Returns:
        DataFrame: Filtered matchups with injury games removed
    """
    if star_players is None:
        star_players = config.STAR_PLAYERS
    for player in star_players:
        game_log = get_game_logs(player, season_year)
        
        for index, row in game_log.iterrows():
            # Convert Timestamp to string format, handling NaT values
            if pd.isna(row['DATE']) or row['DATE'] is None:
                continue  # Skip rows with null dates
            
            try:
                date = row['DATE'].strftime('%Y-%m-%d')
            except (ValueError, AttributeError):
                # If strftime fails, try string conversion
                date = str(row['DATE']).split(' ')[0]
                if date == 'NaT' or date == 'None':
                    continue  # Skip invalid dates

            # Determine home and visitor teams
            home = None
            visitor = None
            if row['HOME/AWAY'] == 'HOME':
                home = row['TEAM']
                visitor = row['OPPONENT']
            else:
                home = row['OPPONENT']
                visitor = row['TEAM']
            
            # Check if player was injured or didn't play
            if row['PTS'] in ['Inactive', 'Did Not Play', 'Did Not Dress']:
                print(f"{player} did not play on {date} due to {row['PTS']}")
                matchups = matchups.drop(
                    matchups[
                        (matchups['DATE'] == date) & 
                        (matchups['HOME'] == get_team_full_name(home)) & 
                        (matchups['VISITOR'] == get_team_full_name(visitor))
                    ].index
                )
            else:
                continue
                
    return matchups


def main():
    """
    Main function to run the NBA prediction testing and analysis.
    """
    print("=== NBA Prediction vs Actual Results Comparison ===")
    
    # Get user input for model selection
    print("Choose prediction model:")
    print("1. Neural Network (original)")
    print("2. Basic Ensemble")
    print("3. Advanced Ensemble")
    model_choice = input("Enter 1, 2, or 3: ").strip()
    
    # Load the selected model
    if model_choice == '2':
        model = load_model('ensemble', season_year)
    elif model_choice == '3':
        model = load_model('advanced', season_year)
    else:
        model = load_model('nn', season_year)
    
    if model is None:
        return
    
    # Get user input for betting options
    print("\nBetting Options:")
    print("1. Interactive betting (manual betting on each game)")
    print("   - You choose bet amount and side for each game")
    print("   - See real-time balance updates")
    print("   - Full control over betting strategy")
    print("2. Prediction accuracy only (no betting)")
    print("   - Focuses purely on prediction accuracy")
    print("   - No betting calculations or financial analysis")
    print("3. Pre-calculate test results (speed up future testing)")
    print("   - Calculate all predictions and store in JSON file")
    print("   - Future testing will be much faster")
    betting_choice = input("Enter 1, 2, or 3: ").strip()
    
    enable_betting = betting_choice == '1'
    
    # Handle pre-calculation option
    if betting_choice == '3':
        print("\nPre-calculating test results...")
        success = pre_calculate_test_results(season_year, model.model_type if hasattr(model, 'model_type') else 'advanced')
        if success:
            print("✅ Test results pre-calculated successfully!")
            print("You can now run option 1 or 2 for faster testing.")
        else:
            print("❌ Failed to pre-calculate test results.")
        return
    
    # Load season data from JSON file (contains both game data and odds)
    season_file = f'data/{season_year}-season.json'
    try:
        with open(season_file, 'r') as f:
            season_data = json.load(f)
        print(f"Loaded {len(season_data)} games from {season_file}")
    except FileNotFoundError:
        print(f"❌ Season file {season_file} not found")
        return
    except Exception as e:
        print(f"❌ Error loading season data: {e}")
        return
    
    # Use season_data as playoff_dataset for odds checking
    playoff_dataset = season_data
    
    # Initialize counters and tracking variables
    total_games = 0
    correct_predictions = 0
    predictions = []
    winnings = 0
    
    # Print header for results table (only when not betting)
    if not enable_betting:
        print("\n" + "="*90)
        print(f"{'Date':<12} {'Away':<4} {'Home':<4} {'Away Pts':<8} {'Home Pts':<8} {'Actual':<8} {'Predicted':<10} {'Prob':<6} {'Correct':<8}")
        print("="*90)
    
    # Process each game from season data
    for game_key, game_data in season_data.items():
        try:
            # Extract game data from season JSON format
            parts = game_key.split('_vs_')
            if len(parts) != 2:
                continue
            
            away_abbr = parts[0]
            home_date_part = parts[1]
            
            # Extract home team and date
            last_underscore = home_date_part.rfind('_')
            if last_underscore == -1:
                continue
            
            home_abbr = home_date_part[:last_underscore]
            date = home_date_part[last_underscore + 1:]
            
            # Check if we have test_results in the season JSON file
            game_key = f"{away_abbr}_vs_{home_abbr}_{date}"
            test_result = None
            
            # Try to load test_results from season data
            try:
                season_file = f'data/{season_year}-season.json'
                with open(season_file, 'r') as f:
                    season_data = json.load(f)
                if game_key in season_data and 'test_result' in season_data[game_key]:
                    test_result = season_data[game_key]['test_result']
                    print(f"Using test_result from season file for {game_key}")
            except:
                pass
            
            if test_result is not None:
                # Use stored test_results
                actual_result = test_result['actual_result']
                visitor_pts = test_result['away_pts']
                home_pts = test_result['home_pts']
            else:
                # Skip games without test_results (no actual game data)
                continue
            
            # Check if betting odds are available for this game
            game_key = f"{away_abbr}_vs_{home_abbr}_{date}"
            if enable_betting and (game_key not in playoff_dataset or 'home_odds' not in playoff_dataset[game_key]):
                # Skip games without odds data when betting is enabled
                continue
            
            # Always calculate prediction in real-time (for betting interface)
            # Get team stats for feature creation
            away_stats = get_team_stats(away_abbr, season_year)
            home_stats = get_team_stats(home_abbr, season_year)
            
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
                    print(f"Applied feature selection: {features_processed.shape[1]} features selected")
                
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
                except (AttributeError, KeyError):
                    # Fallback to standard predict methods
                    features_processed = features.reshape(1, -1)
                    
                    # Scale features if scaler is available and fitted
                    if hasattr(model, 'scaler') and model.scaler is not None:
                        try:
                            features_processed = model.scaler.transform(features_processed)
                        except Exception as e:
                            print(f"Warning: Scaler not fitted properly, using unscaled features: {e}")
                    
                    # Apply feature selection if available
                    if hasattr(model, 'selected_feature_indices') and model.selected_feature_indices is not None:
                        features_processed = features_processed[:, model.selected_feature_indices]
                        print(f"Applied feature selection: {features_processed.shape[1]} features selected")
                    
                    pred = model.predict(features_processed)[0]
                    proba = model.predict_proba(features_processed)
                    probability = proba[0][1] if len(proba[0]) > 1 else proba[0][0]
            else:
                probability = model.predict_probability(features.reshape(1, -1))[0][0]
                pred = int(probability > 0.5)
            
            # Determine if prediction was correct
            correct = (pred == actual_result)
            
            # Interactive betting interface
            current_bet = 0
            bet_side = None
            home_odds = 2.0
            away_odds = 2.0
            
            if enable_betting:
                try:
                    home_odds = playoff_dataset[game_key].get("home_odds", 2.0)
                    away_odds = playoff_dataset[game_key].get("away_odds", 2.0)
                    
                    # Show game info and model prediction (without score)
                    print(f"\n{'='*60}")
                    print(f"GAME: {away_abbr} @ {home_abbr} - {date}")
                    print(f"MODEL PREDICTION: {'Away Win' if pred == 1 else 'Home Win'} (Confidence: {probability:.1%})")
                    print(f"ODDS: {away_abbr} {away_odds:.2f} | {home_abbr} {home_odds:.2f}")
                    print(f"CURRENT BALANCE: ${winnings:.2f}")
                    print(f"{'='*60}")
                    
                    # Get user bet
                    while True:
                        try:
                            bet_input = input("Enter bet amount (or '0' to skip, 'q' to quit): $").strip()
                            if bet_input.lower() == 'q':
                                print("\n" + "="*60)
                                print("QUITTING - FINAL RESULTS")
                                print("="*60)
                                
                                # Calculate final statistics
                                total_games_played = len([p for p in predictions if p.get('bet_amount', 0) > 0])
                                total_bet = sum(p.get('bet_amount', 0) for p in predictions)
                                roi = (winnings / total_bet * 100) if total_bet > 0 else 0
                                
                                print(f"Games Bet On: {total_games_played}")
                                print(f"Total Amount Bet: ${total_bet:.2f}")
                                print(f"Final Balance: ${winnings:.2f}")
                                print(f"Total Profit/Loss: ${winnings:.2f}")
                                print(f"ROI: {roi:.2f}%")
                                
                                if total_games_played > 0:
                                    winning_bets = len([p for p in predictions if p.get('bet_amount', 0) > 0 and 
                                                       ((p.get('bet_side') == 'away' and p.get('actual') == 1) or 
                                                        (p.get('bet_side') == 'home' and p.get('actual') == 0))])
                                    bet_accuracy = (winning_bets / total_games_played * 100) if total_games_played > 0 else 0
                                    print(f"Betting Accuracy: {bet_accuracy:.1f}% ({winning_bets}/{total_games_played})")
                                
                                print("="*60)
                                return
                            elif bet_input == '0':
                                current_bet = 0
                                bet_side = None
                                break
                            else:
                                current_bet = float(bet_input)
                                if current_bet <= 0:
                                    print("Bet amount must be positive!")
                                    continue
                                
                                # Get bet side
                                print(f"Bet on: 1) {away_abbr} (Away) or 2) {home_abbr} (Home)?")
                                side_input = input("Enter 1 or 2: ").strip()
                                if side_input == '1':
                                    bet_side = 'away'
                                    break
                                elif side_input == '2':
                                    bet_side = 'home'
                                    break
                                else:
                                    print("Invalid choice! Enter 1 or 2.")
                                    continue
                        except ValueError:
                            print("Invalid bet amount! Enter a number.")
                            continue
                    
                    # Show the actual result and calculate winnings
                    print(f"\nACTUAL RESULT: {away_abbr} {visitor_pts} - {home_abbr} {home_pts}")
                    
                    # Calculate winnings based on bet
                    if current_bet > 0 and bet_side:
                        if bet_side == 'away' and actual_result == 1:  # Bet away, away won
                            winnings += current_bet * away_odds - current_bet
                            print(f"✓ WIN! +${current_bet * away_odds - current_bet:.2f}")
                        elif bet_side == 'home' and actual_result == 0:  # Bet home, home won
                            winnings += current_bet * home_odds - current_bet
                            print(f"✓ WIN! +${current_bet * home_odds - current_bet:.2f}")
                        else:  # Lost bet
                            winnings -= current_bet
                            print(f"✗ LOSS! -${current_bet:.2f}")
                    else:
                        print("No bet placed")
                    
                    print(f"NEW BALANCE: ${winnings:.2f}")
                except Exception as e:
                    print(f"Error processing betting for {date}: {e}")
                    current_bet = 0
            
            # Update counters
            correct = (pred == actual_result)
            total_games += 1
            if correct:
                correct_predictions += 1
            
            # Store prediction data for analysis
            prediction_data = {
                'date': date,
                'away': away_abbr,
                'home': home_abbr,
                'away_pts': visitor_pts,
                'home_pts': home_pts,
                'actual': actual_result,
                'predicted': pred,
                'probability': probability,
                'correct': correct
            }
            
            if enable_betting:
                prediction_data.update({
                    'bet_amount': current_bet,
                    'bet_side': bet_side,
                    'winnings': winnings
                })
            
            predictions.append(prediction_data)
            
            # Print result for current game (only when not betting)
            if not enable_betting:
                actual_str = "Away" if actual_result == 1 else "Home"
                predicted_str = "Away" if pred == 1 else "Home"
                correct_str = "✓" if correct else "✗"
                print(f"{date:<12} {away_abbr:<4} {home_abbr:<4} {visitor_pts:<8} {home_pts:<8} {actual_str:<8} {predicted_str:<10} {probability:<6.2f} {correct_str:<8}")
            
        except Exception as e:
            print(f"Error processing game {index}: {e}")
            continue
    
    # Calculate and display final results
    accuracy = (correct_predictions / total_games) * 100 if total_games > 0 else 0
    
    # Display final summary
    if enable_betting:
        print("="*120)
    else:
        print("="*90)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total Games Processed: {total_games}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Incorrect Predictions: {total_games - correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if enable_betting:
        # Calculate total amount bet and ROI
        total_bet = sum(p.get('bet_amount', 0) for p in predictions if 'bet_amount' in p)
        roi = (winnings / total_bet * 100) if total_bet > 0 else 0
        
        print(f"Total Amount Bet: ${total_bet:.2f}")
        print(f"Total Winnings: ${winnings:.2f}")
        print(f"ROI: {roi:.2f}%")
    
    # Additional analysis by prediction confidence
    if predictions:
        # Analyze by prediction confidence levels
        high_confidence = [p for p in predictions if p['probability'] > 0.7 or p['probability'] < 0.3]
        medium_confidence = [p for p in predictions if 0.4 <= p['probability'] <= 0.6]
        low_confidence = [p for p in predictions if (0.3 < p['probability'] < 0.4) or (0.6 < p['probability'] < 0.7)]
        
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


if __name__ == "__main__":
    main()
    


