"""
Module containing state of game statistics
"""
from enum import Enum

class BallState(Enum):
    """
    Indicates the status of the ball at any given frame.
    """
    IN_POSSESSION = 1 # hold/dribble
    IN_TRANSITION = 2 # pass/shot
    OUT_OF_PLAY = 3 # out of bounds, just after shot, etc.

class StatState:
    """
    State class holding: player positions, ball position, and team scores
    """
    def __init__(self, output_path):
        # TODO pass in output path text file and initialise fields
        
        # IMMUTABLE
        self.rim = {} # xmin, xmax, ymin, ymax
        self.backboard = {} # coordinates
        self.court_lines = {} # TBD

        # MUTABLE
        self.states = [] # [{'ball': {xmin, xmax, ymin, ymax}, 'playerid'...}]
        self.players = {} # {'player1id': {'shots': 0, "points": 0, "rebounds": 0, "assists": 0}, player2id: ...}
        self.ball_state = [] # [(start_frame, end_frame, BallState)]
        self.passes = {} # {'pass_id': {'frames': (start_frame, end_frame)}, 'players': (p1_id, p2_id)}}
        self.possession = {} # {'player_id': [start_frame, end_frame]}
        self.team1 = [] # [player1, player2]
        self.team2 = []
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
