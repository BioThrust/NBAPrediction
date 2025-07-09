"""
NBA Season Data Combiner

This script combines multiple season datasets into one large training dataset.
This is useful for training models on multiple years of data for better performance.
"""

import json
import sys
import os

def combine_season_data(season_years, output_file=None):
    """
    Combine multiple season datasets into one large dataset.
    
    Args:
        season_years (list): List of season years to combine
        output_file (str, optional): Output file name. If None, uses 'combined-seasons.json'
    
    Returns:
        dict: Combined dataset
    """
    if output_file is None:
        output_file = '../data/combined-seasons.json'
    
    combined_data = {}
    total_games = 0
    
    print(f"Combining data from {len(season_years)} seasons...")
    
    for season_year in season_years:
        season_file = f'../data/{season_year}-season.json'
        
        if not os.path.exists(season_file):
            print(f"Warning: {season_file} not found, skipping...")
            continue
        
        try:
            with open(season_file, 'r') as f:
                season_data = json.load(f)
            
            # Add season prefix to game keys to avoid conflicts
            season_games = 0
            for game_key, game_data in season_data.items():
                new_key = f"{season_year}_{game_key}"
                combined_data[new_key] = game_data
                season_games += 1
            
            total_games += season_games
            print(f"  Season {season_year}: {season_games} games added")
            
        except Exception as e:
            print(f"Error loading {season_file}: {e}")
            continue
    
    # Save combined dataset
    try:
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=4)
        
        print(f"\nCombined dataset saved to {output_file}")
        print(f"Total games: {total_games}")
        print(f"Total seasons: {len(season_years)}")
        
        return combined_data
        
    except Exception as e:
        print(f"Error saving combined dataset: {e}")
        return None

def main():
    """Main function to combine season data."""
    print("=== NBA Season Data Combiner ===")
    
    if len(sys.argv) < 2:
        print("Usage: python combine_seasons.py <season1> <season2> ... [output_file]")
        print("Example: python combine_seasons.py 2022 2023 2024")
        print("Example: python combine_seasons.py 2022 2023 2024 my_combined_data.json")
        sys.exit(1)
    
    # Get season years from command line arguments
    season_years = []
    output_file = None
    
    for arg in sys.argv[1:]:
        if arg.endswith('.json'):
            output_file = f'../data/{arg}'
        else:
            try:
                season_year = int(arg)
                season_years.append(season_year)
            except ValueError:
                print(f"Error: '{arg}' is not a valid season year")
                sys.exit(1)
    
    if not season_years:
        print("Error: No valid season years provided")
        sys.exit(1)
    
    # Sort seasons for consistent ordering
    season_years.sort()
    
    print(f"Combining seasons: {', '.join(map(str, season_years))}")
    if output_file:
        print(f"Output file: {output_file}")
    
    # Combine the data
    combined_data = combine_season_data(season_years, output_file)
    
    if combined_data:
        print("\n✅ Season combination completed successfully!")
        print("\nYou can now use this combined dataset for training:")
        print("1. Update your training scripts to use the combined dataset")
        print("2. Run train_neural.bat or ensemble training")
        print("3. The combined dataset will provide more training examples")
        sys.exit(0)  # Explicitly exit with success code
    else:
        print("\n❌ Season combination failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 