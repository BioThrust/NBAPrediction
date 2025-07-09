"""
NBA Playoff Data Collection and Processing Script

This script collects and processes NBA game data for a specified season, including team statistics,
game schedules, and injury information. It creates a comprehensive dataset for model training
and prediction analysis.
"""

import json
import time
import os
import re
import sys
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
from .reference_scraper import get_team_stats_dict
from .basketball_reference_scraper.seasons import get_schedule
from .basketball_reference_scraper.players import get_game_logs

# Import configuration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SEASON_YEAR, SEASON_DATA_FILE, STAR_PLAYERS, TEAM_FULL_TO_ABBR, get_team_stats_cache_file


def get_team_full_name(abbreviation):
    """
    Convert team abbreviation to full name.
    
    Args:
        abbreviation (str): Team abbreviation (e.g., "BOS")
    
    Returns:
        str: Full team name (e.g., "Boston Celtics")
    """
    from config import TEAM_ABBR_TO_FULL
    return TEAM_ABBR_TO_FULL.get(abbreviation.upper(), abbreviation)


def remove_injuries(matchups, season_year):
    """
    Remove games where either team has a major injury to star players.
    
    Args:
        matchups (DataFrame): DataFrame containing game matchups
        season_year (int): NBA season year
    
    Returns:
        DataFrame: Filtered matchups with injury games removed
    """
    for player in STAR_PLAYERS:
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


def main(season_year=None):
    """
    Main function to collect and process NBA playoff data.
    
    Args:
        season_year (int, optional): NBA season year. If None, uses config default.
    """
    # Use provided season year or default from config
    # print(season_year)
    if season_year is None:
        season_year = SEASON_YEAR
    
    print(f"Collecting data for {season_year-1}-{season_year} NBA season...")
    
    # Get season schedule
    this_year = get_schedule(season_year)
    
    # Remove games with major injuries
    this_year = remove_injuries(this_year, season_year)
    
    # Print columns to see what we have
    print("Available columns:", this_year.columns.tolist())
    
    # Get all unique teams from the schedule
    all_teams = list(TEAM_FULL_TO_ABBR.values())
    
    print("\nFetching stats for all teams...")
    
    # Cache all team stats
    team_stats_cache = {}
    
    # Check if team stats cache already exists for this season
    team_stats_cache_file = get_team_stats_cache_file(season_year)
    if os.path.exists(team_stats_cache_file):
        print(f"Loading existing team stats cache for {season_year} season...")
        try:
            with open(team_stats_cache_file, 'r') as f:
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
            team_stats_cache[team] = get_team_stats_dict(team, season_year)
        print("All team stats cached!")
        
        # Save team stats cache to a separate file
        print(f"Saving team stats cache to {team_stats_cache_file}...")
        with open(team_stats_cache_file, 'w') as f:
            json.dump(team_stats_cache, f, indent=4)
        print("Team stats cache saved successfully!")
    else:
        print("Using existing team stats cache!")
    
    # Initialize the dataset dictionary
    playoff_dataset = {}
    
    # Process the games using cached stats
    print("\nProcessing games...")
    for index, row in this_year.iterrows():
        visitor = TEAM_FULL_TO_ABBR[row['VISITOR']]
        home = TEAM_FULL_TO_ABBR[row['HOME']]
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
    season_data_file = f'data/{season_year}-season.json'
    with open(season_data_file, 'w') as f:
        json.dump(playoff_dataset, f, indent=4)
    print(f"Dataset saved successfully to {season_data_file}!")
    
    # Now scrape odds data for betting analysis
    print("\n" + "="*60)
    print("SCRAPING ODDS DATA FOR TRAINING")
    print("="*60)
    
    # Date mapping for odds portal
    date_dict = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
    }
    
    # Set up Chrome options for odds scraping
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-animations")
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(5)
    
    print("Selenium WebDriver initialized successfully")
    
    # Scrape odds from OddsPortal
    season_display = f"{season_year-1}-{season_year}"
    total_pages = 28  # Adjust based on season length
    
    print(f"Scraping odds for {season_display} season...")
    
    for i in range(total_pages):
        try:
            driver.get(f"https://www.oddsportal.com/basketball/usa/nba-{season_display}/results/#/page/{i+1}/")
            driver.refresh()
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(13)
            
            # Only do dropdown selection once, not on every page
            if i == 0:
                odd_selector = driver.find_element(By.CSS_SELECTOR, '[data-testid="header-odds-formats-selector"]')
                odd_selector.find_element(By.CLASS_NAME, 'drop-arrow').click()
                
                dropdown_options = driver.find_element(By.CLASS_NAME, 'dropdown-content')
                dropdown_options.find_elements(By.CLASS_NAME, 'cursor-pointer')[0].click()
            
            print(f"Processing odds page {i+1}/{total_pages}")
            
            # Aggressive page loading
            body = driver.find_element(By.TAG_NAME, 'body')
            for _ in range(10):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(1)
            
            time.sleep(5)
            
            # Extract page HTML for local parsing
            page_html = driver.page_source
            soup = BeautifulSoup(page_html, 'html.parser')
            
            current_date = None
            game_rows = soup.find_all('div', class_='eventRow')
            print(f"Found {len(game_rows)} game rows on page {i+1}")
            
            # Initialize counters for progress tracking
            total_dataset_games = len(playoff_dataset)
            processed_games = 0
            successful_games = 0
            
            for game_row in game_rows:
                processed_games += 1
                
                try:
                    # Check for date header
                    date_header = game_row.find('div', attrs={'data-testid': 'secondary-header'})
                    if date_header:
                        date_text = date_header.get_text().replace("\n", ' ')
                        current_date = date_text.split(' ')
                    
                    if not current_date or "Pre-season" in current_date or "All" in current_date:
                        continue
                    
                    # Find teams
                    team_elements = game_row.find_all('p', class_='participant-name')
                    if len(team_elements) < 2:
                        continue
                    
                    home_team_name = team_elements[0].get_text().strip()
                    away_team_name = team_elements[1].get_text().strip()
                    
                    # Convert to abbreviations
                    home_team = TEAM_FULL_TO_ABBR.get(home_team_name)
                    away_team = TEAM_FULL_TO_ABBR.get(away_team_name)
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Find odds
                    odds_elements = game_row.find_all(['div', 'p'], attrs={'data-testid': re.compile(r'odd-container-.*')})
                    
                    if len(odds_elements) > 2:
                        result = 0
                        home_odds = float(odds_elements[0].get_text().strip())
                        away_odds = float(odds_elements[2].get_text().strip())
                        if(home_odds<away_odds):
                            result = 0
                        else:
                            result = 1

                        # Create game key
                        game_key = f"{away_team}_vs_{home_team}_{current_date[2]}-{date_dict[current_date[1]]}-{current_date[0]}"
                        
                        # Check if game exists in dataset and add odds
                        if game_key in playoff_dataset:
                            playoff_dataset[game_key]["home_odds"] = home_odds
                            playoff_dataset[game_key]["away_odds"] = away_odds
                            playoff_dataset[game_key]["result"] = result
                            successful_games += 1
                
                except Exception as e:
                    continue
            
            # Progress update for this page
            print(f"Page {i+1} completed: {successful_games} games matched with odds")
            
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
            continue
    
    # Close the driver
    driver.quit()
    
    # Final statistics
    games_with_odds = sum(1 for game in playoff_dataset.values() if 'home_odds' in game)
    print(f"\nOdds scraping completed!")
    print(f"Total games in dataset: {len(playoff_dataset)}")
    print(f"Games with odds data: {games_with_odds}")
    print(f"Odds coverage: {games_with_odds/len(playoff_dataset)*100:.1f}%")
    
    # Save the updated dataset with odds
    print("Saving updated dataset with odds...")
    with open(season_data_file, 'w') as f:
        json.dump(playoff_dataset, f, indent=4)
    print(f"Updated dataset saved to {season_data_file}!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # print(f"Collecting data for {sys.argv[1]} season...")
        season_year = int(sys.argv[1])
    else:
        season_year = SEASON_YEAR
    main(season_year)