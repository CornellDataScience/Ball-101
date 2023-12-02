# results_formatter.py
import json
from ..state import GameState  # Assuming GameState class is in state.py
class Format:
    def results():
        return format_results_for_api(GameState)
def format_results_for_api(game_state: GameState) -> str:
    """
    Takes a GameState object and formats the results into a JSON string for API output.
    Args:
        game_state (GameState): The game state object containing all the game data.
    
    Returns:
        str: A JSON string containing the formatted results.
    """

    # General Game Stats
    general_stats = {
        'Total Frames': len(game_state.frames),
        'Number of Possessions': len(game_state.possessions),
        'Number of Shot Attempts': len(game_state.shot_attempts)
    }

    # Player Stats
    player_stats = {
        player_id: {
            'Total Frames': player_state.frames,
            'Field Goals Attempted': player_state.field_goals_attempted,
            'Field Goals Made': player_state.field_goals,
            'Points Scored': player_state.points,
            'Field Goal Percentage': player_state.field_goal_percentage,
            'Passes': player_state.passes
        } for player_id, player_state in game_state.players.items()
    }

    # Team Stats
    team_stats = {
        'Team 1': {
            'Shots Attempted': game_state.team1.shots_attempted,
            'Shots Made': game_state.team1.shots_made,
            'Points': game_state.team1.points,
            'Field Goal Percentage': game_state.team1.field_goal_percentage
        },
        'Team 2': {
            'Shots Attempted': game_state.team2.shots_attempted,
            'Shots Made': game_state.team2.shots_made,
            'Points': game_state.team2.points,
            'Field Goal Percentage': game_state.team2.field_goal_percentage
        }
    }

    # Ball Stats
    ball_stats = {
        'Ball Frames': game_state.ball.frames
    }

    # Combine all the stats into one dictionary
    results = {
        'General Stats': general_stats,
        'Player Stats': player_stats,
        'Team Stats': team_stats,
        'Ball Stats': ball_stats
    }

    # Convert the dictionary into a JSON string
    results_json = json.dumps(results, indent=4)
    return results_json
