#!/usr/bin/env python3
"""
Simple script to run NBA ensemble predictions
"""

import sys
import json
import os
from ensemble_model import EnsembleNBAPredictor
from advanced_ensemble import AdvancedEnsembleNBAPredictor

def save_ensemble_weights(ensemble, ensemble_type, performance_metrics, dataset_year=None):
    """
    Save ensemble weights in a format compatible with test_results.py and predict_game.py
    
    Args:
        ensemble: Trained ensemble model
        ensemble_type (str): 'basic' or 'advanced'
        performance_metrics (dict): Performance metrics from training
        dataset_year (str): Dataset year used for training (e.g., '2024', 'combined')
    """
    # Defensive checks before saving
    ensemble_weights = ensemble.weights if hasattr(ensemble, 'weights') and ensemble.weights is not None else {}
    feature_names = ensemble.feature_names if hasattr(ensemble, 'feature_names') and ensemble.feature_names is not None else []
    
    weights_data = {
        'model_type': f'ensemble_{ensemble_type}',
        'ensemble_weights': ensemble_weights,
        'is_trained': True,
        'model_performance': performance_metrics,
        'feature_names': feature_names,
        'dataset_year': dataset_year
    }
    
    # Add advanced ensemble specific data
    if ensemble_type == 'advanced':
        weights_data.update({
            'betting_thresholds': ensemble.betting_thresholds if hasattr(ensemble, 'betting_thresholds') else {},
            'meta_model_trained': hasattr(ensemble, 'meta_model') and ensemble.meta_model is not None,
            'voting_models_trained': hasattr(ensemble, 'voting_soft') and ensemble.voting_soft is not None
        })
    
    # Save to data directory with year/combined in filename
    if dataset_year:
        filename = f'../data/{dataset_year}_ensemble_{ensemble_type}_weights.json'
    else:
        filename = f'../data/ensemble_{ensemble_type}_weights.json'
    
    with open(filename, 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    print(f"Ensemble weights saved to {filename}")
    return filename

def run_basic_ensemble(dataset_year=None):
    """Run the basic ensemble model"""
    print("=== Running Basic Ensemble Model ===")
    
    try:
        # Initialize and train basic ensemble
        ensemble = EnsembleNBAPredictor()
        
        # Load data with specific dataset year if provided
        if dataset_year:
            # Temporarily set sys.argv to pass dataset choice to load_data
            original_argv = sys.argv.copy()
            sys.argv = ['run_ensemble.py', 'basic', dataset_year]
            X, y = ensemble.load_data()
            sys.argv = original_argv  # Restore original argv
        else:
        X, y = ensemble.load_data()
            
        ensemble.initialize_models()
        
        # Train models
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        ensemble.train_models(X_train, y_train, X_test, y_test)
        ensemble_acc = ensemble.evaluate_ensemble(X_test, y_test)
        
        # Calculate additional metrics
        from sklearn.metrics import roc_auc_score, classification_report
        y_proba = ensemble.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_proba)
        
        performance_metrics = {
            'mean_accuracy': ensemble_acc * 100,
            'auc_score': auc_score,
            'test_size': len(X_test),
            'training_size': len(X_train)
        }
        
        print(f"\nBasic Ensemble Accuracy: {ensemble_acc:.4f}")
        print(f"Basic Ensemble AUC: {auc_score:.4f}")
        
        # Save ensemble weights
        print("Ensemble weights before saving:", ensemble.weights)
        if not ensemble.weights:
            print("Error: Ensemble weights are empty after training! Not saving.")
            return ensemble
        save_ensemble_weights(ensemble, 'basic', performance_metrics, dataset_year)
        
        return ensemble
        
    except Exception as e:
        print(f"Error running basic ensemble: {e}")
        return None

def run_advanced_ensemble(dataset_year=None):
    """Run the advanced ensemble model with betting analysis"""
    print("\n=== Running Advanced Ensemble Model ===")
    
    try:
        # Initialize and train advanced ensemble
        ensemble = AdvancedEnsembleNBAPredictor()
        
        # Load data with specific dataset year if provided
        if dataset_year:
            # Temporarily set sys.argv to pass dataset choice to load_data
            original_argv = sys.argv.copy()
            sys.argv = ['run_ensemble.py', 'advanced', dataset_year]
            X, y = ensemble.load_data()
            sys.argv = original_argv  # Restore original argv
        else:
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
        
        performance_metrics = {
            'mean_accuracy': acc * 100,
            'auc_score': auc,
            'test_size': len(X_test),
            'training_size': len(X_train),
            'betting_analysis': True
        }
        
        print(f"\nAdvanced Ensemble Accuracy: {acc:.4f}")
        print(f"Advanced Ensemble ROC-AUC: {auc:.4f}")
        
        # Save ensemble weights
        print("Ensemble weights before saving:", ensemble.weights)
        if not ensemble.weights:
            print("Error: Advanced ensemble weights are empty after training! Not saving.")
            return ensemble
        
        # Use default year if dataset_year is None
        save_year = dataset_year if dataset_year else '2024'
        ensemble.save_ensemble(f'../data/{save_year}_ensemble_advanced_weights.json')
        
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
        dataset_year = sys.argv[2] if len(sys.argv) > 2 else None
        
        if command == "basic":
            ensemble = run_basic_ensemble(dataset_year)
            if ensemble and len(sys.argv) >= 4:
                away_team = sys.argv[3]
                home_team = sys.argv[4]
                predict_specific_game(ensemble, away_team, home_team)
                
        elif command == "advanced":
            ensemble = run_advanced_ensemble(dataset_year)
            if ensemble and len(sys.argv) >= 4:
                away_team = sys.argv[3]
                home_team = sys.argv[4]
                odds = None
                if len(sys.argv) >= 7:
                    odds = {'home': float(sys.argv[5]), 'away': float(sys.argv[6])}
                predict_specific_game(ensemble, away_team, home_team, odds)
                
        elif command == "both":
            basic_ensemble = run_basic_ensemble(dataset_year)
            advanced_ensemble = run_advanced_ensemble(dataset_year)
            
            if basic_ensemble and advanced_ensemble and len(sys.argv) >= 4:
                away_team = sys.argv[3]
                home_team = sys.argv[4]
                odds = None
                if len(sys.argv) >= 7:
                    odds = {'home': float(sys.argv[5]), 'away': float(sys.argv[6])}
                
                print("\n" + "="*50)
                predict_specific_game(basic_ensemble, away_team, home_team)
                predict_specific_game(advanced_ensemble, away_team, home_team, odds)
        
        else:
            print("Usage:")
            print("  python run_ensemble.py basic [dataset_year] [away_team] [home_team]")
            print("  python run_ensemble.py advanced [dataset_year] [away_team] [home_team] [home_odds] [away_odds]")
            print("  python run_ensemble.py both [dataset_year] [away_team] [home_team] [home_odds] [away_odds]")
            print("\nExamples:")
            print("  python run_ensemble.py basic 2024 BOS LAL")
            print("  python run_ensemble.py advanced combined BOS LAL 1.85 2.05")
            print("  python run_ensemble.py both 2024 BOS LAL 1.85 2.05")
    else:
        # Interactive mode - only for game prediction, not ensemble type selection
        print("Interactive Mode - Enter teams for prediction:")
                away_team = input("Enter away team (e.g., BOS): ").strip()
                home_team = input("Enter home team (e.g., LAL): ").strip()
        
        # Run both ensembles for comparison
            basic_ensemble = run_basic_ensemble()
            advanced_ensemble = run_advanced_ensemble()
            
            if basic_ensemble and advanced_ensemble:
            use_odds = input("Use odds for advanced ensemble? (y/n): ").strip().lower()
                
                odds = None
                if use_odds == 'y':
                    home_odds = float(input("Enter home odds (e.g., 1.85): "))
                    away_odds = float(input("Enter away odds (e.g., 2.05): "))
                    odds = {'home': home_odds, 'away': away_odds}
                
                print("\n" + "="*50)
                predict_specific_game(basic_ensemble, away_team, home_team)
                predict_specific_game(advanced_ensemble, away_team, home_team, odds)

if __name__ == "__main__":
    main() 