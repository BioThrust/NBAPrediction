#!/usr/bin/env python3
"""
Simple script to run NBA ensemble predictions
"""

import sys
import json
from .ensemble_model import EnsembleNBAPredictor
from .advanced_ensemble import AdvancedEnsembleNBAPredictor

def run_basic_ensemble():
    """Run the basic ensemble model"""
    print("=== Running Basic Ensemble Model ===")
    
    try:
        # Initialize and train basic ensemble
        ensemble = EnsembleNBAPredictor()
        X, y = ensemble.load_data()
        ensemble.initialize_models()
        
        # Train models
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        ensemble.train_models(X_train, y_train, X_test, y_test)
        ensemble_acc = ensemble.evaluate_ensemble(X_test, y_test)
        
        print(f"\nBasic Ensemble Accuracy: {ensemble_acc:.4f}")
        return ensemble
        
    except Exception as e:
        print(f"Error running basic ensemble: {e}")
        return None

def run_advanced_ensemble():
    """Run the advanced ensemble model with betting analysis"""
    print("\n=== Running Advanced Ensemble Model ===")
    
    try:
        # Initialize and train advanced ensemble
        ensemble = AdvancedEnsembleNBAPredictor()
        X, y = ensemble.load_data()
        ensemble.initialize_models()
        
        # Train models
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Split odds data
        odds_train = [ensemble.odds[i] for i in range(len(X)) if i < len(X_train)]
        odds_test = [ensemble.odds[i] for i in range(len(X)) if i >= len(X_train)]
        
        # Train ensembles
        ensemble.train_stacking_ensemble(X_train, y_train, X_test, y_test)
        ensemble.train_voting_ensemble(X_train, y_train)
        ensemble.optimize_betting_thresholds(X_train, y_train, odds_train)
        
        # Evaluate performance
        acc, auc = ensemble.evaluate_ensemble_performance(X_test, y_test, odds_test)
        
        print(f"\nAdvanced Ensemble Accuracy: {acc:.4f}")
        print(f"Advanced Ensemble ROC-AUC: {auc:.4f}")
        
        return ensemble
        
    except Exception as e:
        print(f"Error running advanced ensemble: {e}")
        return None

def predict_specific_game(ensemble, away_team, home_team, odds=None):
    """Predict a specific game, and allow user to continue testing more games interactively."""
    while True:
        print(f"\n=== Predicting {away_team} @ {home_team} ===")
        try:
            if isinstance(ensemble, AdvancedEnsembleNBAPredictor):
                prediction = ensemble.predict_game_advanced(away_team, home_team, odds)
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
            else:
                prediction = ensemble.predict_game(away_team, home_team)
                print(f"Ensemble Prediction: {prediction['ensemble_prediction']} "
                      f"(Probability: {prediction['ensemble_probability']:.3f})")
                print("\nIndividual Model Predictions:")
                for model_name, model_pred in prediction['model_predictions'].items():
                    print(f"  {model_name}: {model_pred['prediction']} "
                          f"(Probability: {model_pred['probability']:.3f})")
        except Exception as e:
            print(f"Error predicting game: {e}")
        # Ask if the user wants to test another game
        again = input("\nTest another game? (y/n): ").strip().lower()
        if again != 'y':
            break
        # Prompt for new teams (and odds if advanced)
        away_team = input("Enter away team (e.g., BOS): ").strip()
        home_team = input("Enter home team (e.g., LAL): ").strip()
        if isinstance(ensemble, AdvancedEnsembleNBAPredictor):
            use_odds = input("Use odds? (y/n): ").strip().lower()
            odds = None
            if use_odds == 'y':
                home_odds = float(input("Enter home odds (e.g., 1.85): "))
                away_odds = float(input("Enter away odds (e.g., 2.05): "))
                odds = {'home': home_odds, 'away': away_odds}

def main():
    """Main function"""
    print("NBA Ensemble Prediction System")
    print("=" * 40)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "basic":
            ensemble = run_basic_ensemble()
            if ensemble and len(sys.argv) >= 4:
                away_team = sys.argv[2]
                home_team = sys.argv[3]
                predict_specific_game(ensemble, away_team, home_team)
                
        elif command == "advanced":
            ensemble = run_advanced_ensemble()
            if ensemble and len(sys.argv) >= 4:
                away_team = sys.argv[2]
                home_team = sys.argv[3]
                odds = None
                if len(sys.argv) >= 6:
                    odds = {'home': float(sys.argv[4]), 'away': float(sys.argv[5])}
                predict_specific_game(ensemble, away_team, home_team, odds)
                
        elif command == "both":
            basic_ensemble = run_basic_ensemble()
            advanced_ensemble = run_advanced_ensemble()
            
            if basic_ensemble and advanced_ensemble and len(sys.argv) >= 4:
                away_team = sys.argv[2]
                home_team = sys.argv[3]
                odds = None
                if len(sys.argv) >= 6:
                    odds = {'home': float(sys.argv[4]), 'away': float(sys.argv[5])}
                
                print("\n" + "="*50)
                predict_specific_game(basic_ensemble, away_team, home_team)
                predict_specific_game(advanced_ensemble, away_team, home_team, odds)
        
        else:
            print("Usage:")
            print("  python run_ensemble.py basic [away_team] [home_team]")
            print("  python run_ensemble.py advanced [away_team] [home_team] [home_odds] [away_odds]")
            print("  python run_ensemble.py both [away_team] [home_team] [home_odds] [away_odds]")
            print("\nExamples:")
            print("  python run_ensemble.py basic BOS LAL")
            print("  python run_ensemble.py advanced BOS LAL 1.85 2.05")
            print("  python run_ensemble.py both BOS LAL 1.85 2.05")
    else:
        # Interactive mode
        print("Interactive Mode - Choose an option:")
        print("1. Run Basic Ensemble")
        print("2. Run Advanced Ensemble")
        print("3. Run Both")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            ensemble = run_basic_ensemble()
            if ensemble:
                away_team = input("Enter away team (e.g., BOS): ").strip()
                home_team = input("Enter home team (e.g., LAL): ").strip()
                predict_specific_game(ensemble, away_team, home_team)
                
        elif choice == "2":
            ensemble = run_advanced_ensemble()
            if ensemble:
                away_team = input("Enter away team (e.g., BOS): ").strip()
                home_team = input("Enter home team (e.g., LAL): ").strip()
                use_odds = input("Use odds? (y/n): ").strip().lower()
                
                odds = None
                if use_odds == 'y':
                    home_odds = float(input("Enter home odds (e.g., 1.85): "))
                    away_odds = float(input("Enter away odds (e.g., 2.05): "))
                    odds = {'home': home_odds, 'away': away_odds}
                
                predict_specific_game(ensemble, away_team, home_team, odds)
                
        elif choice == "3":
            basic_ensemble = run_basic_ensemble()
            advanced_ensemble = run_advanced_ensemble()
            
            if basic_ensemble and advanced_ensemble:
                away_team = input("Enter away team (e.g., BOS): ").strip()
                home_team = input("Enter home team (e.g., LAL): ").strip()
                use_odds = input("Use odds? (y/n): ").strip().lower()
                
                odds = None
                if use_odds == 'y':
                    home_odds = float(input("Enter home odds (e.g., 1.85): "))
                    away_odds = float(input("Enter away odds (e.g., 2.05): "))
                    odds = {'home': home_odds, 'away': away_odds}
                
                print("\n" + "="*50)
                predict_specific_game(basic_ensemble, away_team, home_team)
                predict_specific_game(advanced_ensemble, away_team, home_team, odds)
        
        elif choice == "4":
            print("Goodbye!")
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main() 