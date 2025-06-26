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

# Add necessary paths to system path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_collection')))

# Import custom modules
from data_collection.basketball_reference_scraper.seasons import get_schedule
from data_collection.basketball_reference_scraper.players import get_game_logs
from utils.shared_utils import PredictionNeuralNetwork, get_team_stats, create_comparison_features
from ensemble_models.ensemble_model import EnsembleNBAPredictor
from ensemble_models.advanced_ensemble import AdvancedEnsembleNBAPredictor

# Load playoff dataset with odds information
with open('json_files/2024-season.json', 'r') as f:
    playoff_dataset = json.load(f)


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


def remove_injuries(matchups):
    """
    Remove games where either team has a major injury to star players.
    
    Args:
        matchups (DataFrame): DataFrame containing game matchups
    
    Returns:
        DataFrame: Filtered matchups with injury games removed
    """
    for player in star_players:
        game_log = get_game_logs(player, 2024)
        
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
        model = load_model('ensemble')
    elif model_choice == '3':
        model = load_model('advanced')
    else:
        model = load_model('nn')
    
    if model is None:
        return
    
    # Get user input for betting options
    print("\nBetting Options:")
    print("1. Enable betting calculations and ROI analysis")
    print("   - Simulates betting on games using real odds data")
    print("   - Calculates potential winnings/losses and ROI")
    print("   - Shows bet amounts and betting performance")
    print("   - Requires odds data from json_files/2024-season.json")
    print("2. Disable betting (prediction accuracy only)")
    print("   - Focuses purely on prediction accuracy")
    print("   - No betting calculations or financial analysis")
    print("   - Faster execution and simpler output")
    betting_choice = input("Enter 1 or 2: ").strip()
    
    enable_betting = betting_choice == '1'
    base_bet = 0
    
    if enable_betting:
        print("\nBetting Configuration:")
        print("1. Fixed bet amount")
        print("   - Bet the same amount on every game")
        print("   - Simple and consistent betting strategy")
        print("   - Good for testing with a specific budget")
        print("2. Variable bet based on confidence")
        print("   - Bet more when prediction confidence is high")
        print("   - Bet less when prediction confidence is medium")
        print("   - Risk-adjusted betting strategy")
        bet_type = input("Enter 1 or 2: ").strip()
        
        if bet_type == '1':
            try:
                base_bet = float(input("Enter fixed bet amount ($): "))
            except ValueError:
                print("Invalid amount. Using default $50.")
                base_bet = 50
        else:
            # Variable betting will be handled in the game loop
            base_bet = 0
    
    # Get the schedule data
    print("Loading 2024 season schedule...")
    matchups = get_schedule(2024)
    matchups = remove_injuries(matchups)
    print(f"Loaded {len(matchups)} games")
    
    # Initialize counters and tracking variables
    total_games = 0
    correct_predictions = 0
    predictions = []
    winnings = 0
    
    # Print header for results table
    if enable_betting:
        print("\n" + "="*120)
        print(f"{'Date':<12} {'Away':<4} {'Home':<4} {'Away Pts':<8} {'Home Pts':<8} {'Actual':<8} {'Predicted':<10} {'Prob':<6} {'Bet':<6} {'Correct':<8}")
        print("="*120)
    else:
        print("\n" + "="*90)
        print(f"{'Date':<12} {'Away':<4} {'Home':<4} {'Away Pts':<8} {'Home Pts':<8} {'Actual':<8} {'Predicted':<10} {'Prob':<6} {'Correct':<8}")
        print("="*90)
    
    # Process each game
    for index, row in matchups.iterrows():
        try:
            # Extract game data
            visitor = row['VISITOR']
            home = row['HOME']
            visitor_pts = row['VISITOR_PTS']
            home_pts = row['HOME_PTS']
            date = row['DATE'].strftime('%Y-%m-%d')
            
            # Skip games without scores (future games)
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
            
            # Get team stats for feature creation
            away_stats = get_team_stats(away_abbr)
            home_stats = get_team_stats(home_abbr)
            
            # Create features and make prediction
            features, feature_names = create_comparison_features(away_stats, home_stats)
            
            # Use the selected model for prediction
            if hasattr(model, 'model_type') and model.model_type == 'ensemble':
                pred = model.predict(features.reshape(1, -1))[0]
                probability = model.predict_proba(features.reshape(1, -1))[0]
            elif hasattr(model, 'model_type') and model.model_type == 'advanced':
                result = model.predict_with_confidence(features.reshape(1, -1))
                pred = result['ensemble_prediction'][0]
                probability = result['ensemble_probability'][0]
            else:
                probability = model.predict_probability(features.reshape(1, -1))[0][0]
                pred = int(probability > 0.5)
            
            # Calculate winnings if betting is enabled
            current_bet = 0
            home_odds = 2.0
            away_odds = 2.0
            
            if enable_betting:
                try:
                    game_key = f"{away_abbr}_vs_{home_abbr}_{date}"
                    if game_key in playoff_dataset:
                        home_odds = playoff_dataset[game_key].get("home_odds", 2.0)
                        away_odds = playoff_dataset[game_key].get("away_odds", 2.0)
                        
                        # Determine bet size based on configuration
                        if bet_type == '1':
                            # Fixed bet amount
                            current_bet = base_bet
                        else:
                            # Variable bet based on confidence
                            if probability > 0.7 or probability < 0.3:
                                current_bet = 55  # High confidence
                            elif 0.4 <= probability <= 0.6:
                                current_bet = 40   # Medium confidence
                            else:
                                current_bet = 25   # Low confidence
                        
                        # Calculate winnings
                        if pred == 0 and actual_result == 0:  # Predicted home win, home won
                            winnings += current_bet * home_odds - current_bet
                        elif pred == 1 and actual_result == 1:  # Predicted away win, away won
                            winnings += current_bet * away_odds - current_bet
                        else:  # Wrong prediction
                            winnings -= current_bet
                    else:
                        # No odds data available
                        current_bet = 0
                except Exception as e:
                    print(f"Error calculating winnings for {date}: {e}")
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
                    'base_bet': current_bet,
                    'winnings': winnings
                })
            
            predictions.append(prediction_data)
            
            # Print result for current game
            actual_str = "Away" if actual_result == 1 else "Home"
            predicted_str = "Away" if pred == 1 else "Home"
            correct_str = "✓" if correct else "✗"
            
            if enable_betting:
                print(f"{date:<12} {away_abbr:<4} {home_abbr:<4} {visitor_pts:<8} {home_pts:<8} {actual_str:<8} {predicted_str:<10} {probability:<6.2f} {current_bet:<6} {correct_str:<8}")
            else:
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
        total_bet = sum(p.get('base_bet', 0) for p in predictions if 'base_bet' in p)
        roi = (winnings / total_bet * 100) if total_bet > 0 else 0
        
        print(f"Total Amount Bet: ${total_bet}")
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
    


