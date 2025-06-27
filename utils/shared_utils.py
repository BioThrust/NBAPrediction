"""
Shared Utilities for NBA Prediction Models

This module contains common utility functions and classes used across the NBA prediction system.
It includes feature creation, team statistics retrieval, and neural network prediction functionality.
"""

import json
import numpy as np
import math

def create_comparison_features(away_team_stats, home_team_stats):
    """
    Create enhanced comparison features for NBA game prediction.
    
    This function generates a comprehensive set of features that compare two teams
    across various statistical categories to predict game outcomes.
    
    Args:
        away_team_stats (dict): Statistics for the away team
        home_team_stats (dict): Statistics for the home team
    
    Returns:
        tuple: (comparison_features, feature_names) - Feature array and corresponding names
    """
    comparison_features = []
    feature_names = []
    
    # Basic team comparison features (away - home)
    for feature in away_team_stats.keys():
        if feature in home_team_stats:
            try:
                away_val = float(away_team_stats[feature])
                home_val = float(home_team_stats[feature])
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
        away_net_rating = float(away_team_stats.get('net_rating', 0))
        home_net_rating = float(home_team_stats.get('net_rating', 0))
        net_rating_advantage = away_net_rating - home_net_rating
        comparison_features.append(net_rating_advantage)
        feature_names.append("net_rating_advantage")
        
        # 2. Away team momentum (only positive when away is better)
        away_momentum = max(0, net_rating_advantage)
        comparison_features.append(away_momentum)
        feature_names.append("away_momentum")
        
        # 3. Offensive vs Defensive Matchup
        away_off_rating = float(away_team_stats.get('offensive_rating', 0))
        home_def_rating = float(home_team_stats.get('defensive_rating', 0))
        offensive_advantage = away_off_rating - home_def_rating
        comparison_features.append(offensive_advantage)
        feature_names.append("offensive_advantage")
        
        # 4. Defensive vs Offensive Matchup
        away_def_rating = float(away_team_stats.get('defensive_rating', 0))
        home_off_rating = float(home_team_stats.get('offensive_rating', 0))
        defensive_advantage = home_off_rating - away_def_rating  # Lower is better for away
        comparison_features.append(defensive_advantage)
        feature_names.append("defensive_advantage")
        
        # 5. Shooting Efficiency Advantage
        away_efg = float(away_team_stats.get('efg_pct', 0))
        home_efg = float(home_team_stats.get('efg_pct', 0))
        shooting_advantage = away_efg - home_efg
        comparison_features.append(shooting_advantage)
        feature_names.append("shooting_advantage")
        
        # 6. Pace Advantage (faster pace might favor away team)
        away_pace = float(away_team_stats.get('pace', 0))
        home_pace = float(home_team_stats.get('pace', 0))
        pace_advantage = away_pace - home_pace
        comparison_features.append(pace_advantage)
        feature_names.append("pace_advantage")
        
        # 7. Turnover Advantage (fewer turnovers is better)
        away_tov = float(away_team_stats.get('offensive_tov', 0))
        home_tov = float(home_team_stats.get('offensive_tov', 0))
        tov_advantage = home_tov - away_tov  # Lower away TOV is better
        comparison_features.append(tov_advantage)
        feature_names.append("turnover_advantage")
        
        # 8. Rebounding Advantage
        away_trb = float(away_team_stats.get('trb', 0))
        home_trb = float(home_team_stats.get('trb', 0))
        rebounding_advantage = away_trb - home_trb
        comparison_features.append(rebounding_advantage)
        feature_names.append("rebounding_advantage")
        
        # 9. Assists Advantage (ball movement)
        away_ast = float(away_team_stats.get('ast', 0))
        home_ast = float(home_team_stats.get('ast', 0))
        assists_advantage = away_ast - home_ast
        comparison_features.append(assists_advantage)
        feature_names.append("assists_advantage")
        
        # 10. Opponent Turnover Forcing Advantage
        away_opp_tov = float(away_team_stats.get('opp_offensive_tov', 0))
        home_opp_tov = float(home_team_stats.get('opp_offensive_tov', 0))
        opp_tov_advantage = away_opp_tov - home_opp_tov  # Higher is better (force more TOVs)
        comparison_features.append(opp_tov_advantage)
        feature_names.append("opp_turnover_advantage")
        
        # 11. Opponent Rebounding Defense Advantage
        away_opp_trb = float(away_team_stats.get('opp_trb', 0))
        home_opp_trb = float(home_team_stats.get('opp_trb', 0))
        opp_reb_advantage = home_opp_trb - away_opp_trb  # Lower opponent rebounds is better
        comparison_features.append(opp_reb_advantage)
        feature_names.append("opp_rebounding_advantage")
        
        # 12. Opponent Assists Defense Advantage
        away_opp_ast = float(away_team_stats.get('opp_ast', 0))
        home_opp_ast = float(home_team_stats.get('opp_ast', 0))
        opp_ast_advantage = home_opp_ast - away_opp_ast  # Lower opponent assists is better
        comparison_features.append(opp_ast_advantage)
        feature_names.append("opp_assists_advantage")
        
        # 13. Opponent Shooting Defense Advantage
        away_opp_efg = float(away_team_stats.get('opp_efg_pct', 0))
        home_opp_efg = float(home_team_stats.get('opp_efg_pct', 0))
        opp_shooting_advantage = home_opp_efg - away_opp_efg  # Lower opponent eFG% is better
        comparison_features.append(opp_shooting_advantage)
        feature_names.append("opp_shooting_advantage")
        
        # 14. Overall Efficiency Rating (weighted combination)
        efficiency_rating = (offensive_advantage * 0.4) + (defensive_advantage * -0.3) + (shooting_advantage * 0.3)
        comparison_features.append(efficiency_rating)
        feature_names.append("overall_efficiency")
        
        # 15. Away Team Strength Indicator (when away team is significantly better)
        away_strength = 1.0 if net_rating_advantage > 3.0 else 0.0
        comparison_features.append(away_strength)
        feature_names.append("away_team_strength")
        
        # 16. Close Game Indicator (when teams are evenly matched)
        close_game = 1.0 if abs(net_rating_advantage) < 2.0 else 0.0
        comparison_features.append(close_game)
        feature_names.append("close_game_indicator")
        
        # 17. High Scoring Game Indicator (both teams have high offensive ratings)
        high_scoring = 1.0 if (away_off_rating > 115 and home_off_rating > 115) else 0.0
        comparison_features.append(high_scoring)
        feature_names.append("high_scoring_game")
        
        # 18. Defensive Battle Indicator (both teams have low offensive ratings)
        defensive_battle = 1.0 if (away_off_rating < 110 and home_off_rating < 110) else 0.0
        comparison_features.append(defensive_battle)
        feature_names.append("defensive_battle")
        
    except Exception as e:
        print(f"Error creating enhanced features: {e}")
        # Add zeros for missing features
        comparison_features.extend([0.0] * 18)
        feature_names.extend([
            "net_rating_advantage", "away_momentum", "offensive_advantage", "defensive_advantage",
            "shooting_advantage", "pace_advantage", "turnover_advantage", "rebounding_advantage",
            "assists_advantage", "opp_turnover_advantage", "opp_rebounding_advantage", 
            "opp_assists_advantage", "opp_shooting_advantage", "overall_efficiency",
            "away_team_strength", "close_game_indicator", "high_scoring_game", "defensive_battle"
        ])
    
    return np.array(comparison_features), feature_names

def get_team_stats(team_abbr, season_year=None):
    """
    Get team statistics from cached data.
    
    Args:
        team_abbr (str): Team abbreviation (e.g., "BOS", "LAL")
        season_year (int, optional): NBA season year. Defaults to config.SEASON_YEAR.
    
    Returns:
        dict: Team statistics dictionary
    """
    if season_year is None:
        import config
        season_year = config.SEASON_YEAR
    
    try:
        # Load cached team stats for specific season
        import config
        team_stats_cache_file = config.get_team_stats_cache_file(season_year)
        with open(team_stats_cache_file, 'r') as f:
            team_stats_cache = json.load(f)
        
        if team_abbr in team_stats_cache:
            return team_stats_cache[team_abbr]
        else:
            print(f"Warning: Team {team_abbr} not found in {season_year} cache. Using average stats.")
            # Return average stats for unknown teams
            return {
                'net_rating': 0.0, 
                'offensive_rating': 115.0, 
                'defensive_rating': 115.0, 
                'efg_pct': 0.55, 
                'pace': 98.0, 
                'offensive_tov': 1100, 
                'trb': 3600, 
                'ast': 2200, 
                'opp_offensive_tov': 13.0, 
                'opp_trb': 42.0, 
                'opp_ast': 25.0, 
                'opp_efg_pct': 0.55
            }
    except FileNotFoundError:
        print(f"Warning: {team_stats_cache_file} not found for {season_year} season. Please run data_collection/playoff_data.py first to generate team stats.")
        print("Using placeholder stats for all teams.")
        # Return average stats as fallback
        return {
            'net_rating': 0.0, 
            'offensive_rating': 115.0, 
            'defensive_rating': 115.0, 
            'efg_pct': 0.55, 
            'pace': 98.0, 
            'offensive_tov': 1100, 
            'trb': 3600, 
            'ast': 2200, 
            'opp_offensive_tov': 13.0, 
            'opp_trb': 42.0, 
            'opp_ast': 25.0, 
            'opp_efg_pct': 0.55
        }

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

class PredictionNeuralNetwork:
    """
    Neural Network class for making predictions using pre-trained weights.
    
    This class loads pre-trained neural network weights and provides methods
    for making predictions on new data.
    """
    
    def __init__(self, weights_data):
        """
        Initialize the neural network with pre-trained weights.
        
        Args:
            weights_data (dict): Dictionary containing neural network weights and architecture
        """
        self.weights1 = np.array(weights_data['weights1'])
        self.weights2 = np.array(weights_data['weights2'])
        self.bias1 = np.array(weights_data['bias1'])
        self.bias2 = np.array(weights_data['bias2'])
        self.input_size = weights_data['input_size']
        self.hidden_size = weights_data['hidden_size']
        self.output_size = weights_data['output_size']
    
    def forward(self, X):
        """
        Forward pass through the neural network.
        
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
    
    def predict_probability(self, X):
        """
        Get probability predictions for input data.
        
        Args:
            X (np.array): Input features
        
        Returns:
            np.array: Probability predictions
        """
        return self.forward(X)
    
    def predict(self, X):
        """
        Make binary predictions for input data.
        
        Args:
            X (np.array): Input features
        
        Returns:
            np.array: Binary predictions (0 or 1)
        """
        probabilities = self.predict_probability(X)
        return (probabilities > 0.5).astype(int) 