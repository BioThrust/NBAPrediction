"""
Betting Odds Accuracy Analysis

This module analyzes betting odds accuracy by scraping historical odds data
and comparing it with actual game outcomes.
"""

import selenium
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import json
import re

# Team name dictionary for converting full names to abbreviations
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

# Date dictionary for converting month names to numbers
date_dict = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12"
}


def get_chrome_driver():
    """
    Create and configure ChromeDriver instance.
    
    Returns:
        webdriver.Chrome: Configured ChromeDriver instance
    """
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-data-dir=/tmp/chrome-data")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")  # Disable images for speed
    chrome_options.add_argument("--disable-javascript")  # Disable JS if not needed
    chrome_options.add_argument("--disable-css")  # Disable CSS if not needed
    chrome_options.add_argument("--disable-animations")  # Disable animations
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(5)  # Reduced from 10 to 5 seconds
    return driver


def analyze_betting_odds():
    """
    Analyze betting odds accuracy by scraping historical data.
    """
    # Load dataset
    with open('json_files/2024-season.json', 'r') as f:
        playoff_dataset = json.load(f)
    
    # Initialize driver
    driver = get_chrome_driver()
    print("Selenium WebDriver initialized successfully")
    
    correct = 0
    total = 0
    winnings = 0
    base_bet = 100
    
    try:
        for i in range(28):
            driver.get(f"https://www.oddsportal.com/basketball/usa/nba-2023-2024/results/#/page/{i+1}/")
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
            
            print(f"Navigated to Historical Odds Page {i+1}")
            
            # Aggressive page loading with multiple techniques
            
            # Technique 1: Use keyboard navigation
            body = driver.find_element(By.TAG_NAME, 'body')
            for _ in range(10):  # Press Page Down 10 times
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(1)

            # Wait a bit more to ensure all lazy-loaded content is fully rendered
            time.sleep(5)
            
            # Extract the entire page HTML once and parse locally
            print("Extracting page HTML for local parsing...")
            page_html = driver.page_source
            soup = BeautifulSoup(page_html, 'html.parser')
            
            games = driver.find_elements(By.CLASS_NAME, 'eventRow')
            current_date = None
            
            # Find all game rows
            game_rows = soup.find_all('div', class_='eventRow')
            print(f"Found {len(game_rows)} game rows in HTML")
            
            # Initialize counters for progress tracking
            total_dataset_games = len(playoff_dataset)
            total_games = len(game_rows)
            processed_games = 0
            successful_games = 0
            
            print(f"Processing page games against {total_dataset_games} total games in dataset")
            
            for game_row in game_rows:
                total += 1
                processed_games += 1
                progress_percent = (processed_games / total_games) * 100
                
                # Create progress bar
                bar_length = 30
                filled_length = int(bar_length * processed_games // total_games)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"\rProgress: [{bar}] {progress_percent:.1f}% ({processed_games}/{total_games}) - Page {i+1}/28", end='', flush=True)
                
                try:
                    winner = game_row.find('div', attrs={'data-testid': 'odd-container-winning'}).get_text().strip()
                    loser = game_row.find('div', attrs={'data-testid': 'odd-container-default'}).get_text().strip()
                    print(winner, loser)
                    if(float(winner) < float(loser)):
                        correct += 1
                        winnings += base_bet * winner - base_bet
                    else:
                        winnings -= base_bet

                except Exception as e:
                    print(f"\nError processing game: {e}")
                    continue
            
            # Final progress update for this page
            print(correct/total*100, "%")
            print(winnings)
            print(f"\nPage {i+1} completed! Processed {processed_games} games, successfully matched {successful_games} games")
            print(f"Total dataset games with results: {sum(1 for game in playoff_dataset.values() if 'result' in game)}/{total_dataset_games}")
    
    finally:
        # Always close the driver
        driver.quit()
        print("ChromeDriver closed successfully")


if __name__ == "__main__":
    analyze_betting_odds()
