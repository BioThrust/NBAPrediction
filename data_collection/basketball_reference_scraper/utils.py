from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import unicodedata, unidecode

try:
    from .request_utils import get_wrapper
except:
    from basketball_reference_scraper.request_utils import get_wrapper

def get_game_suffix(date, team1, team2):
    r = get_wrapper(f'https://www.basketball-reference.com/boxscores/?month={date.month}&year={date.year}&day={date.day}')
    if r.status_code==200:
        soup = BeautifulSoup(r.content, 'html.parser')
        for table in soup.find_all('table', attrs={'class': 'teams'}):
            for anchor in table.find_all('a'):
                if 'boxscores' in anchor.attrs['href']:
                    if team1 in str(anchor.attrs['href']) or team2 in str(anchor.attrs['href']):
                        suffix = anchor.attrs['href']
                        return suffix
"""
    Helper function for inplace creation of suffixes--necessary in order
    to fetch rookies and other players who aren't in the /players
    catalogue. Added functionality so that players with abbreviated names
    can still have a suffix created.
"""
def create_last_name_part_of_suffix(potential_last_names):
    last_names = ''.join(potential_last_names)
    if len(last_names) <= 5:
        return last_names[:].lower()
    else:
        return last_names[:5].lower()

"""
    Amended version of the original suffix function--it now creates all
    suffixes in place.

    Since basketball reference standardizes URL codes, it is much more efficient
    to create them locally and compare names to the page results. The maximum
    amount of times a player code repeats is 5, but only 2 players have this
    problem--meaning most player URLs are correctly accessed within 1 to 2
    iterations of the while loop below.

    Added unidecode to make normalizing incoming string characters more
    consistent.

    This implementation dropped player lookup fail count from 306 to 35 to 0.
"""
def get_player_suffix(name):
    normalized_name = unidecode.unidecode(unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8"))
    if normalized_name == 'Metta World Peace' :
        suffix = '/players/a/artesro01.html'
    else:
        split_normalized_name = normalized_name.split(' ')
        if len(split_normalized_name) < 2:
            return None
        initial = normalized_name.split(' ')[1][0].lower()
        all_names = name.split(' ')
        # print(all_names)
        first_name_part = unidecode.unidecode(all_names[0][:2].lower())
        first_name = all_names[0]
        other_names = all_names[1:]

        last_name_part = create_last_name_part_of_suffix(other_names)
        suffix = '/players/'+initial+'/'+last_name_part+first_name_part+'01'

   
    return suffix


def remove_accents(name, team, season_end_year):
    alphabet = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXZY ')
    if len(set(name).difference(alphabet))==0:
        return name
    
    # Use Selenium as default
    from .request_utils import get_selenium_wrapper
    table_html = get_selenium_wrapper(f'https://www.basketball-reference.com/teams/{team}/{season_end_year}.html', "//table")
    
    team_df = None
    best_match = name
    
    if table_html:
        team_df = pd.read_html(table_html)[0]
    else:
        print("No team table found")
        return name
    
    if team_df is not None:
        max_matches = 0
        for p in team_df['Player']:
            matches = sum(l1 == l2 for l1, l2 in zip(p, name))
            if matches>max_matches:
                max_matches = matches
                best_match = p
    return best_match
