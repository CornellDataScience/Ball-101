"""
Module containing state of game statistics
"""

class StatState:
    """
    State class holding player positions, ball position, and team scores
    """
    def __init__(self):
        self.players = [] # each player: dict of features
        self.ball_position = (0, 0)
        self.score_1 = 0
        self.score_2 = 0


    # Dummy function
    def inc_score(self, team, value):
        """
        Updates the scores of the team that scored.
        Parameters: 
        team (int): 1 or 2.
        value (int): 2 or 3.

        """
        assert team in (0, 1) and value in (2,3)

        if team == 1:
            self.score_1 += value
        else:
            self.score_2 += value
