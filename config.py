"""
NBA Prediction Configuration

This file contains global configuration settings for the NBA prediction project.
Modify these values to change the season, file paths, and other settings.
"""

# =============================================================================
# SEASON CONFIGURATION
# =============================================================================

# NBA Season to analyze (e.g., 2024 for 2023-2024 season, 2025 for 2024-2025 season)
SEASON_YEAR = 2024

# Season display format for URLs and file names
# For 2024 season: "2023-2024" (Basketball Reference format)
# For 2025 season: "2024-2025" (Basketball Reference format)
SEASON_DISPLAY = f"{SEASON_YEAR-1}-{SEASON_YEAR}"

# =============================================================================
# FILE PATHS
# =============================================================================

# Main dataset file
SEASON_DATA_FILE = f"json_files/{SEASON_YEAR}-season.json"

# Model weights file
WEIGHTS_FILE = "json_files/weights.json"

# Team stats cache file (season-specific)
def get_team_stats_cache_file(season_year):
    """Get the team stats cache file path for a specific season."""
    return f"json_files/{season_year}_team_stats_cache.json"

# Default team stats cache file (for backward compatibility)
TEAM_STATS_CACHE_FILE = get_team_stats_cache_file(SEASON_YEAR)

# Ensemble model weights file
ENSEMBLE_WEIGHTS_FILE = "json_files/ensemble_weights.json"

# =============================================================================
# ODDS PORTAL CONFIGURATION
# =============================================================================

# Odds Portal URL format for the season
ODDS_PORTAL_URL = f"https://www.oddsportal.com/basketball/usa/nba-{SEASON_DISPLAY}/results/"

# =============================================================================
# BETTING CONFIGURATION
# =============================================================================

# Default bet amounts for variable betting strategy
HIGH_CONFIDENCE_BET = 55    # Bet when probability > 0.7 or < 0.3
MEDIUM_CONFIDENCE_BET = 40  # Bet when 0.4 <= probability <= 0.6
LOW_CONFIDENCE_BET = 25     # Bet when 0.3 < probability < 0.4 or 0.6 < probability < 0.7

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Neural network training parameters
NEURAL_NET_EPOCHS = 1000
NEURAL_NET_LEARNING_RATE = 0.01
NEURAL_NET_BATCH_SIZE = 32

# Cross-validation settings
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# DATA COLLECTION SETTINGS
# =============================================================================

# Star players to monitor for injuries (used for injury filtering)
STAR_PLAYERS = [
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

# =============================================================================
# TEAM MAPPINGS
# =============================================================================

# Team abbreviation to full name mapping
TEAM_ABBR_TO_FULL = {
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

# Full name to abbreviation mapping
TEAM_FULL_TO_ABBR = {v: k for k, v in TEAM_ABBR_TO_FULL.items()} 