"""
Module containing state of game statistics
"""
from enum import Enum
from team_detect import team_split


class BallState(Enum):
    """
    Indicates the status of the ball at any given frame.
    """
    IN_POSSESSION = 1  # hold/dribble
    IN_TRANSITION = 2  # pass/shot
    OUT_OF_PLAY = 3  # out of bounds, just after shot, etc.


class StatState:
    """
    State class holding: player positions, ball position, and team scores
    """

    def __init__(self, output_path):
        # TODO pass in output path text file and initialise fields
        rim, frames = self.parse_output(output_path)

        # IMMUTABLE
        self.rim = rim  # xmin, xmax, ymin, ymax
        self.backboard = {}  # coordinates
        self.court_lines = {}  # TBD

        # MUTABLE
        # [{'ball': {xmin, xmax, ymin, ymax}, 'playerid'...}]
        self.states = frames
        # {'player1id': {'shots': 0, "points": 0, "rebounds": 0, "assists": 0},
        # player2id: ...}
        self.players = {}
        # [(start_frame, end_frame, BallState)]
        self.ball_state = []
        # {'pass_id': {'frames': (start_frame, end_frame)},
        # 'players':(p1_id, p2_id)}}
        self.passes = {}
        # {'player_id': [start_frame, end_frame]}
        self.possession = {}
        # [player1, player2]
        self.team1 = []
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
        assert team in (0, 1) and value in (2, 3)

        if team == 1:
            self.score_1 += value
        else:
            self.score_2 += value

    def parse_output(self, output_path):
        """
        Parses the output file into a list of dictionaries.
        File is in the format:
        Each line represents a frame and contains the following information:
            <frame_no> <obj_type> <obj_id> <xmin>, <ymin>, <width>, <height>,
            <irrelevant>
        Object types: 0 - ball, 1 - player, 2 - rim
        Input:
            output_path [str]: path to output file
        Output:
            rim_info [dict]: dictionary containing rim coordinates
            frames [list]: list of dictionaries containing frame information
        """
        frames = []
        rim_info = {}
        with open(output_path, 'r') as f:
            lines = f.readlines()
            curr_frame = lines[0].split()[0]
            frame_info = {}
            rim = True
            for line in lines:
                curr = line.split()
                if curr_frame != curr[0]:
                    frames.append(frame_info)
                    frame_info = {}
                    curr_frame = curr[0]
                if curr[1] == '0':
                    frame_info['ball'] = {
                        'xmin': curr[3],
                        'ymin': curr[4],
                        'xmax': curr[3] + curr[5],
                        'ymax': curr[4] + curr[6]
                    }
                elif curr[1] == '1':
                    frame_info['player' + curr[2]] = {
                        'xmin': curr[3],
                        'ymin': curr[4],
                        'xmax': curr[3] + curr[5],
                        'ymax': curr[4] + curr[6]
                    }
                elif rim and curr[1] == '2':
                    rim_info = {
                        'xmin': curr[3],
                        'ymin': curr[4],
                        'xmax': curr[3] + curr[5],
                        'ymax': curr[4] + curr[6]
                    }
                    rim = False
            frames.append(frame_info)
        return rim_info, frames
