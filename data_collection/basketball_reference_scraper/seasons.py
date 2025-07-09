"""
Basketball Reference Season Data Scraper

This module provides functions to scrape NBA season schedules and standings from Basketball Reference.
It handles regular season and playoff data, with special handling for the 2020 COVID-affected season.
"""

import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

try:
    from .request_utils import get_wrapper
except:
    from basketball_reference_scraper.request_utils import get_wrapper


def get_schedule(season, playoffs=False):
    """
    Get NBA schedule for a specific season.
    
    Args:
        season (int): NBA season year (e.g., 2024)
        playoffs (bool): Whether to include playoff games
    
    Returns:
        pd.DataFrame: DataFrame containing schedule with columns:
            - DATE: Game date
            - VISITOR: Away team name
            - VISITOR_PTS: Away team points
            - HOME: Home team name
            - HOME_PTS: Home team points
    """
    # Define months for the season
    months = ['October', 'November', 'December', 'January', 'February', 'March',
              'April', 'May', 'June']
    
    # Special handling for 2020 COVID-affected season
    if season == 2020:
        months = ['October-2019', 'November', 'December', 'January', 'February', 'March',
                  'July', 'August', 'September', 'October-2020']
    
    df = pd.DataFrame()
    
    # Use Selenium as default for all scraping
    from .request_utils import get_selenium_wrapper
    
    # Scrape each month's schedule
    for month in months:
        try:
            print(f"Scraping {month} schedule for {season} season...")
            
            url = f'https://www.basketball-reference.com/leagues/NBA_{season}_games-{month.lower()}.html'
            table_html = get_selenium_wrapper(url, "//table[@id='schedule']")
            
            if table_html:
                month_df = pd.read_html(table_html)[0]
                df = pd.concat([df, month_df])
                print(f"Successfully scraped {len(month_df)} games from {month}")
            else:
                print(f"No schedule table found for {month}")
        except Exception as e:
            print(f"Error scraping {month} schedule: {e}")
            continue

    # Check if we got any data
    if df.empty:
        print(f"ERROR: No schedule data was successfully scraped for {season} season!")
        print("This could be due to:")
        print("1. Rate limiting from Basketball Reference")
        print("2. Changes in the website structure")
        print("3. Network connectivity issues")
        print("4. The season data may not be available yet")
        raise ValueError(f"No schedule data available for {season} season")

    # Clean up the DataFrame
    df = df.reset_index()
    
    # Remove unnecessary columns
    cols_to_remove = [i for i in df.columns if 'Unnamed' in i]
    cols_to_remove += [i for i in df.columns if 'Notes' in i]
    cols_to_remove += [i for i in df.columns if 'Start' in i]
    cols_to_remove += [i for i in df.columns if 'Attend' in i]
    cols_to_remove += [i for i in df.columns if 'Arena' in i]
    cols_to_remove += [i for i in df.columns if 'LOG' in i]
    cols_to_remove += ['index']
    df = df.drop(cols_to_remove, axis=1)
    
    # Clean the data - remove header rows and invalid entries
    print(f"Cleaning data... Original shape: {df.shape}")
    
    # Remove rows where DATE column contains header text
    df = df[~df.iloc[:, 0].str.contains('Date', case=False, na=False)]
    df = df[~df.iloc[:, 0].str.contains('GAME', case=False, na=False)]
    df = df[~df.iloc[:, 0].str.contains('HOME', case=False, na=False)]
    
    # Remove rows with NaN or empty values in DATE column
    df = df.dropna(subset=[df.columns[0]])
    df = df[df.iloc[:, 0].str.strip() != '']
    
    print(f"After cleaning: {df.shape}")
    
    # Rename columns for consistency
    df.columns = ['DATE', 'VISITOR', 'VISITOR_PTS', 'HOME', 'HOME_PTS']

    # Handle 2020 season special case
    if season == 2020:
        df = df[df['DATE'] != 'Playoffs']
        df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x))
        df = df.sort_values(by='DATE')
        df = df.reset_index().drop('index', axis=1)
        
        # Find playoff start date
        playoff_loc = df[df['DATE'] == pd.to_datetime('2020-08-17')].head(n=1)
        if len(playoff_loc.index) > 0:
            playoff_index = playoff_loc.index[0]
        else:
            playoff_index = len(df)
        
        # Filter based on playoffs parameter
        if playoffs:
            df = df[playoff_index:]
        else:
            df = df[:playoff_index]
    else:
        # Handle regular seasons
        # Special handling for 1953 season with multiple playoff headers
        if season == 1953:
            df.drop_duplicates(subset=['DATE', 'HOME', 'VISITOR'], inplace=True)
        
        # Find playoff start
        playoff_loc = df[df['DATE'] == 'Playoffs']
        if len(playoff_loc.index) > 0:
            playoff_index = playoff_loc.index[0]
        else:
            playoff_index = len(df)
        
        # Filter based on playoffs parameter
        if playoffs:
            df = df[playoff_index + 1:]
        else:
            df = df[:playoff_index]
        
        # Convert dates to datetime with error handling
        def safe_datetime_convert(x):
            try:
                return pd.to_datetime(x)
            except:
                print(f"Warning: Could not parse date '{x}', skipping...")
                return pd.NaT
        
        df['DATE'] = df['DATE'].apply(safe_datetime_convert)
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['DATE'])
    
    print(f"Successfully processed {len(df)} games for {season} season")
    return df


def get_standings(date=None):
    """
    Get NBA standings for a specific date.
    
    Args:
        date (datetime, optional): Date to get standings for. Defaults to current date.
    
    Returns:
        dict: Dictionary containing Eastern and Western conference standings DataFrames
            - EASTERN_CONF: Eastern conference standings
            - WESTERN_CONF: Western conference standings
    
    Raises:
        ConnectionError: If request to Basketball Reference fails
    """
    if date is None:
        date = datetime.now()
    else:
        date = pd.to_datetime(date)
    
    d = {}
    
    # Scrape standings page
    r = get_wrapper(f'https://www.basketball-reference.com/friv/standings.fcgi?month={date.month}&day={date.day}&year={date.year}')
    
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Find conference tables
        e_table = soup.find('table', attrs={'id': 'standings_e'})
        w_table = soup.find('table', attrs={'id': 'standings_w'})
        
        # Initialize DataFrames
        e_df = pd.DataFrame(columns=['TEAM', 'W', 'L', 'W/L%', 'GB', 'PW', 'PL', 'PS/G', 'PA/G'])
        w_df = pd.DataFrame(columns=['TEAM', 'W', 'L', 'W/L%', 'GB', 'PW', 'PL', 'PS/G', 'PA/G'])
        
        # Parse Eastern Conference standings
        if e_table:
            e_df = pd.read_html(str(e_table))[0]
            e_df.rename(columns={'Eastern Conference': 'TEAM'}, inplace=True)
        
        # Parse Western Conference standings
        if w_table:
            w_df = pd.read_html(str(w_table))[0]
            w_df.rename(columns={'Western Conference': 'TEAM'}, inplace=True)
        
        d['EASTERN_CONF'] = e_df
        d['WESTERN_CONF'] = w_df
        
        return d
    else:
        raise ConnectionError('Request to basketball reference failed')
