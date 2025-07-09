from .basketball_reference_scraper.teams import get_roster, get_all_team_stats, get_roster_stats
from .basketball_reference_scraper.box_scores import get_box_scores

# print(get_team_misc('BOS', 2024)['ORtg'])

def get_team_stats_dict(team_abbr, season):
    # Get all team stats in one call
    team_stats, opp_stats, team_misc = get_all_team_stats(team_abbr, season)
    
    return {
        # Team's own stats
        'net_rating': team_misc['ORtg'] - team_misc['DRtg'],
        'offensive_rating': team_misc['ORtg'],
        'defensive_rating': team_misc['DRtg'],
        'efg_pct': team_misc['eFG%'].values[0],
        'pace': team_misc['Pace'],
        'offensive_tov': team_stats['TOV'],
        'trb': team_stats['TRB'],
        'ast': team_stats['AST'],
        
        # Opponent stats
        'opp_offensive_tov': opp_stats['TOV'],
        'opp_trb': opp_stats['TRB'],
        'opp_ast': opp_stats['AST'],
        'opp_efg_pct': team_misc['eFG%'].values[1]
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


