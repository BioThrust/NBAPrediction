from .basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc
from .basketball_reference_scraper.box_scores import get_box_scores

# print(get_team_misc('BOS', 2024)['ORtg'])

def get_team_stats_dict(team_abbr, season):
    # Get team's own stats from both sources
    team_stats = get_team_misc(team_abbr, season)
    team_basic = get_team_stats(team_abbr, season)
    
    # Get opponent stats from both sources
    opp_stats = get_opp_stats(team_abbr, season)
    opp_basic = get_opp_stats(team_abbr, season)
    return {
        # Team's own stats
        'net_rating': team_stats['ORtg'] - team_stats['DRtg'],
        'offensive_rating': team_stats['ORtg'],
        'defensive_rating': team_stats['DRtg'],
        'efg_pct': team_stats['eFG%'].values[0],
        'pace': team_stats['Pace'],
        'offensive_tov': team_basic['TOV'],
        'trb': team_basic['TRB'],
        'ast': team_basic['AST'],
        
        # Opponent stats
        'opp_offensive_tov': opp_basic['TOV'],
        'opp_trb': opp_basic['TRB'],
        'opp_ast': opp_basic['AST'],
        'opp_efg_pct': team_stats['eFG%'].values[1]
    }

def create_game_dict(team1, team2, season, game_date): # Ended up not using this function
    # Get stats for both teams
    team1_stats = get_team_stats_dict(team1, season)
    team2_stats = get_team_stats_dict(team2, season)
    
    # Create the game dictionary
    game_dict = {
        f"{team1}_vs_{team2}_{game_date}": {
            team1: team1_stats,
            team2: team2_stats
        }
    }
    
    return game_dict


