"""
NBA Playoff Data Collection and Processing Script

This script collects and processes NBA game data for the 2024 season, including team statistics,
game schedules, and injury information. It creates a comprehensive dataset for model training
and prediction analysis.
"""

import json
import time
import os
import re
from datetime import datetime

import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

# Import custom modules
from data_collection.reference_scraper import get_team_stats_dict
from .basketball_reference_scraper.seasons import get_schedule
from .basketball_reference_scraper.players import get_game_logs


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

            print(date)

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
    Main function to collect and process NBA playoff data.
    """
    # Get 2024 season schedule
    this_year = get_schedule(2024)
    
    # Remove games with major injuries
    this_year = remove_injuries(this_year)
    
    # Print columns to see what we have
    print("Available columns:", this_year.columns.tolist())
    
    # Team name mapping dictionary
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
    
    # Get all unique teams from the schedule
    all_teams = list(team_name_dict.values())
    
    print("\nFetching stats for all teams...")
    
    # Cache all team stats
    team_stats_cache = {}
    
    # Check if team stats cache already exists
    if os.path.exists('json_files/team_stats_cache.json'):
        print("Loading existing team stats cache...")
        try:
            with open('json_files/team_stats_cache.json', 'r') as f:
                team_stats_cache = json.load(f)
            print(f"Successfully loaded cached stats for {len(team_stats_cache)} teams!")
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Will fetch fresh stats...")
            team_stats_cache = {}
    
    # Only fetch stats if cache is empty
    if not team_stats_cache:
        print("Fetching fresh team stats...")
        for team in all_teams:
            print(f"Fetching stats for {team}...")
            team_stats_cache[team] = get_team_stats_dict(team, 2024)
        print("All team stats cached!")
        
        # Save team stats cache to a separate file
        print("Saving team stats cache to team_stats_cache.json...")
        with open('json_files/team_stats_cache.json', 'w') as f:
            json.dump(team_stats_cache, f, indent=4)
        print("Team stats cache saved successfully!")
    else:
        print("Using existing team stats cache!")
    
    # Initialize the dataset dictionary
    playoff_dataset = {}
    
    # Process the games using cached stats
    print("\nProcessing games...")
    for index, row in this_year.iterrows():
        visitor = team_name_dict[row['VISITOR']]
        home = team_name_dict[row['HOME']]
        date = row['DATE'].strftime('%Y-%m-%d')
        visitor_score = row['VISITOR_PTS']
        home_score = row['HOME_PTS']
        
        # Create game dictionary with cached stats
        game_key = f"{visitor}_vs_{home}_{date}"
        game_dict = {
            game_key: {
                visitor: team_stats_cache[visitor],
                home: team_stats_cache[home]
            }
        }
        
        playoff_dataset.update(game_dict)
    
    # Save the processed dataset
    print("Saving processed dataset...")
    with open('json_files/2024-season.json', 'w') as f:
        json.dump(playoff_dataset, f, indent=4)
    print("Dataset saved successfully!")


if __name__ == "__main__":
    main() 
