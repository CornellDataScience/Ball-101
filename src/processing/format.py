# results_formatter.py
from ..state import GameState  # Adjust the import according to your project structure

class Format:
    @staticmethod
    def results() -> str:
        game_state= GameState
        """
        Takes a GameState object and formats its contents into plain text, then returns the text content.

        Args:
            game_state (GameState): The game state object containing all game data.

        Returns:
            str: The content of the formatted results.
        """
        # Format the results into a plain text string
        results_str = "General Game Stats:\n"
        results_str += f"Total Frames: {len(game_state.frames)}\n"
        results_str += f"Number of Possessions: {len(game_state.possessions)}\n"
        results_str += f"Number of Shot Attempts: {len(game_state.shot_attempts)}\n\n"

        # Format Player Stats
        results_str += "Player Stats:\n"
        for player_id, player_state in game_state.players.items():
            results_str += f"Player {player_id}:\n"
            results_str += f"  Total Frames: {player_state.frames}\n"
            results_str += f"  Field Goals Attempted: {player_state.field_goals_attempted}\n"
            results_str += f"  Field Goals Made: {player_state.field_goals}\n"
            results_str += f"  Points Scored: {player_state.points}\n"
            results_str += f"  Field Goal Percentage: {player_state.field_goal_percentage}\n"
            results_str += f"  Passes: {player_state.passes}\n\n"

        # Format Team Stats
        results_str += "Team Stats:\n"
        for team_id, team_state in [('Team 1', game_state.team1), ('Team 2', game_state.team2)]:
            results_str += f"{team_id}:\n"
            results_str += f"  Shots Attempted: {team_state.shots_attempted}\n"
            results_str += f"  Shots Made: {team_state.shots_made}\n"
            results_str += f"  Points: {team_state.points}\n"
            results_str += f"  Field Goal Percentage: {team_state.field_goal_percentage}\n\n"

        # Format Ball Stats
        results_str += "Ball Stats:\n"
        results_str += f"Ball Frames: {game_state.ball.frames}\n"

        # Return the results string
        return results_str
