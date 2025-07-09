"""
Shared Utilities for NBA Prediction Models

This module contains common utility functions and classes used across the NBA prediction system.
It includes feature creation, team statistics retrieval, and neural network prediction functionality.
"""

import json
import numpy as np
import math
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def select_best_features(X, y, k=None, method='mutual_info'):
    """
    Select the best features using statistical tests.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target labels
        k (int): Number of features to select (if None, select top 80%)
        method (str): Feature selection method ('mutual_info', 'f_classif', 'random_forest')
    
    Returns:
        tuple: (X_selected, selected_indices, feature_scores)
    """
    if k is None:
        k = max(1, int(X.shape[1] * 0.8))  # Select top 80% of features
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    elif method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'random_forest':
        # Use Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importance = rf.feature_importances_
        selected_indices = np.argsort(feature_importance)[-k:]
        return X[:, selected_indices], selected_indices, feature_importance
    else:
        raise ValueError("Method must be 'mutual_info', 'f_classif', or 'random_forest'")
    
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    return X_selected, selected_indices, feature_scores

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
        
        # NEW ENHANCED FEATURES FOR BETTER ACCURACY
        
        # 19. Advanced Efficiency Metrics
        # True Shooting Percentage equivalent (using eFG% as proxy)
        away_ts_pct = away_efg * 1.1  # Approximate TS% from eFG%
        home_ts_pct = home_efg * 1.1
        ts_advantage = away_ts_pct - home_ts_pct
        comparison_features.append(ts_advantage)
        feature_names.append("true_shooting_advantage")
        
        # 20. Possession Efficiency (Pace-adjusted scoring)
        away_poss_efficiency = away_off_rating / (away_pace + 1e-8)
        home_poss_efficiency = home_off_rating / (home_pace + 1e-8)
        poss_efficiency_advantage = away_poss_efficiency - home_poss_efficiency
        comparison_features.append(poss_efficiency_advantage)
        feature_names.append("possession_efficiency_advantage")
        
        # 21. Defensive Intensity (lower opponent shooting + higher opponent turnovers)
        away_def_intensity = (home_opp_efg * -1) + (away_opp_tov * 0.1)
        home_def_intensity = (away_opp_efg * -1) + (home_opp_tov * 0.1)
        def_intensity_advantage = away_def_intensity - home_def_intensity
        comparison_features.append(def_intensity_advantage)
        feature_names.append("defensive_intensity_advantage")
        
        # 22. Ball Movement Quality (assists per possession)
        away_ast_per_poss = away_ast / (away_pace + 1e-8)
        home_ast_per_poss = home_ast / (home_pace + 1e-8)
        ast_per_poss_advantage = away_ast_per_poss - home_ast_per_poss
        comparison_features.append(ast_per_poss_advantage)
        feature_names.append("assists_per_possession_advantage")
        
        # 23. Rebounding Efficiency (rebounds per possession)
        away_reb_per_poss = away_trb / (away_pace + 1e-8)
        home_reb_per_poss = home_trb / (home_pace + 1e-8)
        reb_per_poss_advantage = away_reb_per_poss - home_reb_per_poss
        comparison_features.append(reb_per_poss_advantage)
        feature_names.append("rebounds_per_possession_advantage")
        
        # 24. Turnover Rate (turnovers per possession)
        away_tov_rate = away_tov / (away_pace + 1e-8)
        home_tov_rate = home_tov / (home_pace + 1e-8)
        tov_rate_advantage = home_tov_rate - away_tov_rate  # Lower is better
        comparison_features.append(tov_rate_advantage)
        feature_names.append("turnover_rate_advantage")
        
        # 25. Opponent Turnover Rate Forcing
        away_opp_tov_rate = away_opp_tov / (away_pace + 1e-8)
        home_opp_tov_rate = home_opp_tov / (home_pace + 1e-8)
        opp_tov_rate_advantage = away_opp_tov_rate - home_opp_tov_rate
        comparison_features.append(opp_tov_rate_advantage)
        feature_names.append("opp_turnover_rate_advantage")
        
        # 26. Net Rating Squared (captures extreme advantages)
        net_rating_squared = net_rating_advantage ** 2
        comparison_features.append(net_rating_squared)
        feature_names.append("net_rating_squared")
        
        # 27. Offensive Rating Interaction (offense vs defense matchup strength)
        off_def_interaction = offensive_advantage * defensive_advantage
        comparison_features.append(off_def_interaction)
        feature_names.append("offensive_defensive_interaction")
        
        # 28. Pace-Shooting Interaction (how pace affects shooting efficiency)
        pace_shooting_interaction = pace_advantage * shooting_advantage
        comparison_features.append(pace_shooting_interaction)
        feature_names.append("pace_shooting_interaction")
        
        # 29. Home Court Adjusted Net Rating
        home_court_adjusted_net = net_rating_advantage - 3.0  # Subtract typical home court advantage
        comparison_features.append(home_court_adjusted_net)
        feature_names.append("home_court_adjusted_net_rating")
        
        # 30. Dominance Indicator (when one team is clearly superior)
        dominance = 1.0 if abs(net_rating_advantage) > 5.0 else 0.0
        comparison_features.append(dominance)
        feature_names.append("dominance_indicator")
        
        # 31. Balanced Team Indicator (both teams have similar offensive/defensive ratings)
        away_balance = abs(away_off_rating - away_def_rating)
        home_balance = abs(home_off_rating - home_def_rating)
        balance_advantage = home_balance - away_balance  # Lower balance is better
        comparison_features.append(balance_advantage)
        feature_names.append("team_balance_advantage")
        
        # 32. Efficiency Gap (difference in efficiency between teams)
        away_efficiency = away_off_rating - away_def_rating
        home_efficiency = home_off_rating - home_def_rating
        efficiency_gap = away_efficiency - home_efficiency
        comparison_features.append(efficiency_gap)
        feature_names.append("efficiency_gap")
        
        # 33. Shooting vs Defense Mismatch
        shooting_defense_mismatch = shooting_advantage - opp_shooting_advantage
        comparison_features.append(shooting_defense_mismatch)
        feature_names.append("shooting_defense_mismatch")
        
        # 34. Turnover vs Defense Mismatch
        turnover_defense_mismatch = tov_advantage - opp_tov_advantage
        comparison_features.append(turnover_defense_mismatch)
        feature_names.append("turnover_defense_mismatch")
        
        # 35. Rebounding vs Defense Mismatch
        rebounding_defense_mismatch = rebounding_advantage - opp_reb_advantage
        comparison_features.append(rebounding_defense_mismatch)
        feature_names.append("rebounding_defense_mismatch")
        
        # 36. Assists vs Defense Mismatch
        assists_defense_mismatch = assists_advantage - opp_ast_advantage
        comparison_features.append(assists_defense_mismatch)
        feature_names.append("assists_defense_mismatch")
        
        # NEW ADVANCED FEATURES FOR SIGNIFICANT ACCURACY IMPROVEMENT
        
        # 37. Recent Form Indicator (using net rating as proxy for recent performance)
        away_recent_form = max(0, away_net_rating)  # Only positive form
        home_recent_form = max(0, home_net_rating)
        recent_form_advantage = away_recent_form - home_recent_form
        comparison_features.append(recent_form_advantage)
        feature_names.append("recent_form_advantage")
        
        # 38. Momentum Indicator (teams on winning streaks vs losing streaks)
        away_momentum = 1.0 if away_net_rating > 2.0 else (0.5 if away_net_rating > 0 else 0.0)
        home_momentum = 1.0 if home_net_rating > 2.0 else (0.5 if home_net_rating > 0 else 0.0)
        momentum_advantage = away_momentum - home_momentum
        comparison_features.append(momentum_advantage)
        feature_names.append("momentum_advantage")
        
        # 39. Elite Team Indicator (teams with very high net ratings)
        away_elite = 1.0 if away_net_rating > 5.0 else 0.0
        home_elite = 1.0 if home_net_rating > 5.0 else 0.0
        elite_advantage = away_elite - home_elite
        comparison_features.append(elite_advantage)
        feature_names.append("elite_team_advantage")
        
        # 40. Tanking Team Indicator (teams with very low net ratings)
        away_tanking = 1.0 if away_net_rating < -5.0 else 0.0
        home_tanking = 1.0 if home_net_rating < -5.0 else 0.0
        tanking_advantage = home_tanking - away_tanking  # Tanking is bad
        comparison_features.append(tanking_advantage)
        feature_names.append("tanking_team_advantage")
        
        # 41. Offensive Firepower (high scoring teams)
        away_firepower = 1.0 if away_off_rating > 115 else 0.0
        home_firepower = 1.0 if home_off_rating > 115 else 0.0
        firepower_advantage = away_firepower - home_firepower
        comparison_features.append(firepower_advantage)
        feature_names.append("offensive_firepower_advantage")
        
        # 42. Defensive Wall (excellent defensive teams)
        away_defense = 1.0 if away_def_rating < 110 else 0.0  # Lower is better
        home_defense = 1.0 if home_def_rating < 110 else 0.0
        defense_advantage = away_defense - home_defense
        comparison_features.append(defense_advantage)
        feature_names.append("defensive_wall_advantage")
        
        # 43. Efficiency Gap Squared (captures extreme efficiency differences)
        efficiency_gap_squared = efficiency_gap ** 2
        comparison_features.append(efficiency_gap_squared)
        feature_names.append("efficiency_gap_squared")
        
        # 44. Net Rating Cubed (captures very extreme advantages)
        net_rating_cubed = net_rating_advantage ** 3
        comparison_features.append(net_rating_cubed)
        feature_names.append("net_rating_cubed")
        
        # 45. Shooting vs Defense Interaction
        shooting_defense_interaction = shooting_advantage * defensive_advantage
        comparison_features.append(shooting_defense_interaction)
        feature_names.append("shooting_defense_interaction")
        
        # 46. Pace vs Efficiency Interaction
        pace_efficiency_interaction = pace_advantage * efficiency_gap
        comparison_features.append(pace_efficiency_interaction)
        feature_names.append("pace_efficiency_interaction")
        
        # 47. Home Court Adjusted Efficiency
        home_court_adjusted_efficiency = efficiency_gap - 3.0  # Subtract home court advantage
        comparison_features.append(home_court_adjusted_efficiency)
        feature_names.append("home_court_adjusted_efficiency")
        
        # 48. Balanced vs Unbalanced Matchup
        away_balance = abs(away_off_rating - away_def_rating)
        home_balance = abs(home_off_rating - home_def_rating)
        balance_difference = away_balance - home_balance
        comparison_features.append(balance_difference)
        feature_names.append("team_balance_difference")
        
        # 49. Extreme Mismatch Indicator (when one team is much better in all areas)
        extreme_mismatch = 1.0 if (abs(net_rating_advantage) > 8.0 and abs(shooting_advantage) > 0.05) else 0.0
        comparison_features.append(extreme_mismatch)
        feature_names.append("extreme_mismatch_indicator")
        
        # 50. Close Game High Scoring (both teams good offensively, close net ratings)
        close_high_scoring = 1.0 if (abs(net_rating_advantage) < 3.0 and away_off_rating > 115 and home_off_rating > 115) else 0.0
        comparison_features.append(close_high_scoring)
        feature_names.append("close_high_scoring_game")
        
        # 51. Defensive Battle Close Game (both teams good defensively, close net ratings)
        defensive_battle_close = 1.0 if (abs(net_rating_advantage) < 3.0 and away_def_rating < 112 and home_def_rating < 112) else 0.0
        comparison_features.append(defensive_battle_close)
        feature_names.append("defensive_battle_close_game")
        
        # 52. Offensive vs Defensive Style Clash
        offensive_defensive_clash = 1.0 if ((away_off_rating > 115 and home_def_rating < 110) or (home_off_rating > 115 and away_def_rating < 110)) else 0.0
        comparison_features.append(offensive_defensive_clash)
        feature_names.append("offensive_defensive_style_clash")
        
        # 53. Similar Style Teams (both offensive or both defensive)
        similar_style = 1.0 if ((away_off_rating > 115 and home_off_rating > 115) or (away_def_rating < 110 and home_def_rating < 110)) else 0.0
        comparison_features.append(similar_style)
        feature_names.append("similar_style_teams")
        
        # 54. Net Rating Percentile (relative strength)
        net_rating_percentile = net_rating_advantage / 20.0  # Normalize to -1 to 1 range
        comparison_features.append(net_rating_percentile)
        feature_names.append("net_rating_percentile")
        
        # 55. Shooting Efficiency Percentile
        shooting_percentile = shooting_advantage / 0.1  # Normalize to reasonable range
        comparison_features.append(shooting_percentile)
        feature_names.append("shooting_efficiency_percentile")
        
        # 56. Defensive Intensity Percentile
        defensive_percentile = defensive_advantage / 10.0  # Normalize
        comparison_features.append(defensive_percentile)
        feature_names.append("defensive_intensity_percentile")
        
        # 57. Overall Team Quality Score
        away_quality = (away_off_rating + (120 - away_def_rating)) / 2  # Higher is better
        home_quality = (home_off_rating + (120 - home_def_rating)) / 2
        quality_advantage = away_quality - home_quality
        comparison_features.append(quality_advantage)
        feature_names.append("overall_team_quality_advantage")
        
        # 58. Matchup Difficulty (how hard the game will be for away team)
        matchup_difficulty = home_quality - away_quality  # Higher = harder for away
        comparison_features.append(matchup_difficulty)
        feature_names.append("matchup_difficulty_for_away")
        
        # 59. Game Type Classification (blowout, close, upset potential)
        if abs(net_rating_advantage) > 8.0:
            game_type = 1.0  # Blowout potential
        elif abs(net_rating_advantage) < 3.0:
            game_type = 0.0  # Close game
        else:
            game_type = 0.5  # Competitive
        comparison_features.append(game_type)
        feature_names.append("game_type_classification")
        
        # 60. Upset Potential (when underdog has some advantages)
        upset_potential = 1.0 if (net_rating_advantage < -3.0 and shooting_advantage > 0.02) else 0.0
        comparison_features.append(upset_potential)
        feature_names.append("upset_potential_indicator")
        
    except Exception as e:
        print(f"Error creating enhanced features: {e}")
        # Add zeros for missing features
        comparison_features.extend([0.0] * 60)
        feature_names.extend([
            "net_rating_advantage", "away_momentum", "offensive_advantage", "defensive_advantage",
            "shooting_advantage", "pace_advantage", "turnover_advantage", "rebounding_advantage",
            "assists_advantage", "opp_turnover_advantage", "opp_rebounding_advantage", 
            "opp_assists_advantage", "opp_shooting_advantage", "overall_efficiency",
            "away_team_strength", "close_game_indicator", "high_scoring_game", "defensive_battle",
            "true_shooting_advantage", "possession_efficiency_advantage", "defensive_intensity_advantage",
            "assists_per_possession_advantage", "rebounds_per_possession_advantage", "turnover_rate_advantage",
            "opp_turnover_rate_advantage", "net_rating_squared", "offensive_defensive_interaction",
            "pace_shooting_interaction", "home_court_adjusted_net_rating", "dominance_indicator",
            "team_balance_advantage", "efficiency_gap", "shooting_defense_mismatch", "turnover_defense_mismatch",
            "rebounding_defense_mismatch", "assists_defense_mismatch", "recent_form_advantage",
            "momentum_advantage", "elite_team_advantage", "tanking_team_advantage", "offensive_firepower_advantage",
            "defensive_wall_advantage", "efficiency_gap_squared", "net_rating_cubed", "shooting_defense_interaction",
            "pace_efficiency_interaction", "home_court_adjusted_efficiency", "team_balance_difference",
            "extreme_mismatch_indicator", "close_high_scoring_game", "defensive_battle_close_game",
            "offensive_defensive_style_clash", "similar_style_teams", "net_rating_percentile",
            "shooting_efficiency_percentile", "defensive_intensity_percentile", "overall_team_quality_advantage",
            "matchup_difficulty_for_away", "game_type_classification", "upset_potential_indicator"
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
        print(f"Warning: {team_stats_cache_file} not found for {season_year} season. Please run data_collection/data_scraper_main.py first to generate team stats.")
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