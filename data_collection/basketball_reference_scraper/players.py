"""
Basketball Reference Player Data Scraper

This module provides functions to scrape NBA player statistics and game logs from Basketball Reference.
It includes functions for getting player stats, game logs, headshots, and splits data.
"""

import pandas as pd
from requests import get
from bs4 import BeautifulSoup

try:
    from .utils import get_player_suffix
    from .lookup import lookup
    from .request_utils import get_wrapper, get_selenium_wrapper
except:
    from basketball_reference_scraper.utils import get_player_suffix
    from basketball_reference_scraper.request_utils import get_wrapper, get_selenium_wrapper
    from basketball_reference_scraper.lookup import lookup

def get_stats(_name, stat_type='PER_GAME', playoffs=False, career=False, ask_matches=True):
    """
    Get player statistics from Basketball Reference.
    
    Args:
        _name (str): Player name
        stat_type (str): Type of stats to retrieve ('PER_GAME', 'TOTALS', 'ADVANCED', 'PER_MINUTE', 'PER_POSS')
        playoffs (bool): Whether to get playoff stats
        career (bool): Whether to get career stats
        ask_matches (bool): Whether to ask for player name matches
    
    Returns:
        pd.DataFrame: DataFrame containing player statistics
    
    Raises:
        ConnectionError: If request to Basketball Reference fails
    """
    name = lookup(_name, ask_matches)
    suffix = get_player_suffix(name)
    
    # Handle case where player suffix is not found
    if not suffix:
        print(f"Could not find player suffix for '{name}'")
        return pd.DataFrame()
    
    stat_type = stat_type.lower()
    table = None
    
    # Use Selenium as default for all stat types
    from .request_utils import get_selenium_wrapper
    
    # Handle different stat types
    if stat_type in ['per_game', 'totals', 'advanced'] and not playoffs:
        xpath = f"//table[@id='{stat_type}']"
        table = get_selenium_wrapper(f'https://www.basketball-reference.com/{suffix}', xpath)
    elif stat_type in ['per_minute', 'per_poss'] or playoffs:
        if playoffs:
            xpath = f"//table[@id='playoffs_{stat_type}']"
        else:
            xpath = f"//table[@id='{stat_type}']"
        table = get_selenium_wrapper(f'https://www.basketball-reference.com/{suffix}', xpath)
    
    if table is None:
        return pd.DataFrame()
    
    # Parse the table
    df = pd.read_html(table)[0]
    
    # Rename columns for consistency
    df.rename(columns={'Season': 'SEASON', 'Age': 'AGE',
                       'Tm': 'TEAM', 'Lg': 'LEAGUE', 'Pos': 'POS', 'Awards': 'AWARDS'}, inplace=True)
    
    # Handle special column names
    if 'FG.1' in df.columns:
        df.rename(columns={'FG.1': 'FG%'}, inplace=True)
    if 'eFG' in df.columns:
        df.rename(columns={'eFG': 'eFG%'}, inplace=True)
    if 'FT.1' in df.columns:
        df.rename(columns={'FT.1': 'FT%'}, inplace=True)

    # Filter based on career parameter
    career_index = df[df['SEASON'] == 'Career'].index[0]
    if career:
        df = df.iloc[career_index + 2:, :]
    else:
        df = df.iloc[:career_index, :]

    df = df.reset_index().drop('index', axis=1)
    return df


def get_game_logs(_name, year, playoffs=False, ask_matches=True):
    """
    Get player game logs for a specific year.
    
    Args:
        _name (str): Player name
        year (int): Season year
        playoffs (bool): Whether to get playoff game logs
        ask_matches (bool): Whether to ask for player name matches
    
    Returns:
        pd.DataFrame: DataFrame containing game logs with columns:
            - DATE: Game date
            - AGE: Player age
            - TEAM: Player's team
            - HOME/AWAY: Whether game was home or away
            - OPPONENT: Opposing team
            - RESULT: Game result (W/L)
            - GAME_SCORE: Game score
            - SERIES: Playoff series (if applicable)
            - PTS: Points scored
            - And other game statistics
    
    Raises:
        ConnectionError: If request to Basketball Reference fails
    """
    # name = lookup(_name, ask_matches)
    suffix = get_player_suffix(_name)
    
    # Handle case where player suffix is not found
    if suffix is None:
        print(f"Could not find player suffix for '{name}'")
        return pd.DataFrame()
    
   
    
    # Determine URL and selector based on playoffs parameter
    if playoffs:
        selector = 'player_game_log_post'
        url = f'https://www.basketball-reference.com/{suffix}/gamelog-playoffs'
    else:
        selector = 'player_game_log_reg'
        url = f'https://www.basketball-reference.com/{suffix}/gamelog/{year}'
    
    # Use Selenium as default
    from .request_utils import get_selenium_wrapper
    xpath = f"//table[@id='{selector}']"
    table_html = get_selenium_wrapper(url, xpath)
    
    if table_html is None:
        print("No suitable table found")
        return pd.DataFrame()
    
    # Parse the game logs table
    df = pd.read_html(table_html)[0]
    
    # Rename columns for consistency
    df.rename(columns={'Date': 'DATE', 'Age': 'AGE', 'Team': 'TEAM', 
                      'Unnamed: 5': 'HOME/AWAY', 'Opp': 'OPPONENT',
                      'Unnamed: 7': 'RESULT', 'GmSc': 'GAME_SCORE', 
                      'Series': 'SERIES'}, inplace=True)
    
    # Convert HOME/AWAY column
    df['HOME/AWAY'] = df['HOME/AWAY'].apply(lambda x: 'AWAY' if x == '@' else 'HOME')
    
    # Remove header rows and unnecessary columns
    df = df[df['Rk'] != 'Rk']
    df = df.drop(['Rk', 'Gtm'], axis=1).reset_index(drop=True)
    
    # Convert date column to datetime (only for regular season)
    if not playoffs:
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    return df


def get_player_headshot(_name, ask_matches=True):
    """
    Get URL for player headshot image.
    
    Args:
        _name (str): Player name
        ask_matches (bool): Whether to ask for player name matches
    
    Returns:
        str: URL to player headshot image
    """
    name = lookup(_name, ask_matches)
    suffix = get_player_suffix(name)
    jpg = suffix.split('/')[-1].replace('html', 'jpg')
    url = 'https://d2cwpp38twqe55.cloudfront.net/req/202006192/images/players/' + jpg
    return url


def get_player_splits(_name, season_end_year, stat_type='PER_GAME', ask_matches=True):
    """
    Get player splits data for a specific season.
    
    Args:
        _name (str): Player name
        season_end_year (int): Season end year (e.g., 2024 for 2023-24 season)
        stat_type (str): Type of stats ('PER_GAME', 'SHOOTING', 'ADVANCED', 'TOTALS')
        ask_matches (bool): Whether to ask for player name matches
    
    Returns:
        pd.DataFrame: DataFrame containing player splits data
    
    Raises:
        ConnectionError: If request to Basketball Reference fails
        Exception: If invalid stat_type is provided
    """
    name = lookup(_name, ask_matches)
    suffix = get_player_suffix(name)[:-5]
    
    r = get_wrapper(f'https://www.basketball-reference.com/{suffix}/splits/{season_end_year}')
    
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, 'html.parser')
        table = soup.find('table')
        
        if table:
            df = pd.read_html(str(table))[0]
            
            # Handle missing split values
            for i in range(1, len(df['Unnamed: 0_level_0', 'Split'])):
                if isinstance(df['Unnamed: 0_level_0', 'Split'][i], float):
                    df['Unnamed: 0_level_0', 'Split'][i] = df['Unnamed: 0_level_0', 'Split'][i - 1]
            
            # Remove total rows
            df = df[~df['Unnamed: 1_level_0', 'Value'].str.contains('Total|Value')]
            
            # Extract headers
            headers = df.iloc[:, :2]
            headers = headers.droplevel(0, axis=1)
            
            # Handle different stat types
            if stat_type.lower() in ['per_game', 'shooting', 'advanced', 'totals']:
                if stat_type.lower() == 'per_game':
                    df = df['Per Game']
                    df['Split'] = headers['Split']
                    df['Value'] = headers['Value']
                    cols = df.columns.tolist()
                    cols = cols[-2:] + cols[:-2]
                    df = df[cols]
                    return df
                elif stat_type.lower() == 'shooting':
                    df = df['Shooting']
                    df['Split'] = headers['Split']
                    df['Value'] = headers['Value']
                    cols = df.columns.tolist()
                    cols = cols[-2:] + cols[:-2]
                    df = df[cols]
                    return df
                elif stat_type.lower() == 'advanced':
                    df = df['Advanced']
                    df['Split'] = headers['Split']
                    df['Value'] = headers['Value']
                    cols = df.columns.tolist()
                    cols = cols[-2:] + cols[:-2]
                    df = df[cols]
                    return df
                elif stat_type.lower() == 'totals':
                    df = df['Totals']
                    df['Split'] = headers['Split']
                    df['Value'] = headers['Value']
                    cols = df.columns.tolist()
                    cols = cols[-2:] + cols[:-2]
                    df = df[cols]
                    return df
            else:
                raise Exception('The "stat_type" you entered does not exist. The following options are: PER_GAME, SHOOTING, ADVANCED, TOTALS')
    else:
        raise ConnectionError('Request to basketball reference failed')