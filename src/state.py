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

    def __init__(self, output_path, player_init_frame, ballFile):
        # TODO pass in output path text file and initialise fields
        rim, frames = self.__parse_output(output_path)
        # IMMUTABLE
        self.rim = rim  # xmin, xmax, ymin, ymax
        self.backboard = {}  # coordinates
        self.court_lines = {}  # TBD

        # MUTABLE
        # [{'ball': {xmin, xmax, ymin, ymax}, 'playerid'...}]
        self.states = frames
        self.__parse_new(ballFile)
        # {'player1id': {'shots': 0, "points": 0, "rebounds": 0, "assists": 0},
        # player2id: ...}
        start_stats = {'shots': 0, "points": 0, "rebounds": 0, "assists": 0}
        init_frame = frames[player_init_frame]
        self.players = {}
        for id in init_frame.keys():
            if id != 'ball':
                self.players[id] = start_stats
        self.playerids = list(self.players.keys())
        teams, pos_lst = team_split(self.playerids, frames)
        # [(start_frame, end_frame, BallState)]
        self.ball_state = self.__ball_state_update(
            pos_lst, (len(frames)-1))
        # {'pass_id': {'frames': (start_frame, end_frame)},
        # 'players':(p1_id, p2_id)}}
        self.passes = self.__player_passes(pos_lst)
        # {'player_id': [(start_frame, end_frame), ...]}
        self.possession = self.__player_possession(pos_lst)
        # [player1, player2]
        self.team1 = teams[0]
        self.team2 = teams[1]
        self.score_1 = 0
        self.score_2 = 0

    def __parse_output(self, output_path):
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
            rim = True
            frame_info = {"frameno": curr_frame, "players": {}}
            for line in lines:
                curr = line.split()
                if curr_frame != curr[0]:
                    frames.append(frame_info)
                    frame_info = {"frameno": curr[0], "players": {}}
                    curr_frame = curr[0]
                if curr[1] == '0':
                    frame_info['ball'] = {
                        'xmin': curr[3],
                        'ymin': curr[4],
                        'xmax': curr[3] + curr[5],
                        'ymax': curr[4] + curr[6]
                    }
                elif curr[1] == '1':
                    frame_info['players']['player' + curr[2]] = {
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

    def __parse_new(self, new_out):
        with open(new_out, 'r') as f:
            lines = f.readlines()
            curr_frame = lines[0].split()[0]
            idx = 0
            for i, frame in enumerate(self.states):
                if frame.get("frameno") == curr_frame:
                    idx = i
                    break
            frame_info = {}
            for line in lines:
                curr = line.split()
                if curr_frame != curr[0]:
                    self.states[idx].update(frame_info)
                    for i in range(idx, len(self.states)):
                        if self.states[i].get("frameno") == curr[0]:
                            idx = i
                            break
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
                    frame_info['players']['player' + curr[2]] = {
                        'xmin': curr[3],
                        'ymin': curr[4],
                        'xmax': curr[3] + curr[5],
                        'ymax': curr[4] + curr[6]
                    }
                elif rim and curr[1] == '2':
                    self.rim = {
                        'xmin': curr[3],
                        'ymin': curr[4],
                        'xmax': curr[3] + curr[5],
                        'ymax': curr[4] + curr[6]
                    }
                    rim = False
        pass

    def __player_passes(self, pos_lst):
        """
        Input:
            pos_lst [list]: list of ball possession tuples
        Output:
            passes [list[tuple]]: Returns a list of passes with each pass
                                represented as a tuple of the form
                                (pass_id, passerid, receiverid,
                                start_frame, end_frame)
        """
        passes = []
        curr_player = pos_lst[0][0]
        curr_end_frame = pos_lst[0][2]
        for i in range(1, len(pos_lst)):
            curr_pass = (i, curr_player, pos_lst[i][0], curr_end_frame+1,
                         pos_lst[i][1]-1)
            passes.append(curr_pass)
            curr_player = pos_lst[i][0]
            curr_end_frame = pos_lst[i][2]
        return passes

    def __ball_state_update(self, pos_lst, lastframe):
        # [(start_frame, end_frame, BallState)]
        ball_state = []
        if pos_lst[0][1] != 0:
            ball_state.append((0, pos_lst[0][1]-1, BallState.OUT_OF_PLAY))
        ball_state.append(
            (pos_lst[0][1], pos_lst[0][2], BallState.IN_POSSESSION))
        curr_frame = pos_lst[0][2]+1
        for i in range(1, len(pos_lst)):
            ball_state.append(
                (curr_frame, pos_lst[i][1]-1, BallState.IN_TRANSITION))
            ball_state.append(
                (pos_lst[i][1], pos_lst[i][2], BallState.IN_POSSESSION))
            curr_frame = pos_lst[i][2]+1
        ball_state.append((curr_frame, lastframe, BallState.OUT_OF_PLAY))
        return ball_state

    def __print_coords(self, coords, tab=0):
        """
        Returns the string coords of an object in the format:
        xmin, xmax, ymin, ymax
        """
        if len(coords) == 0:
            return "NONE"
        xmin = "xmin: " + str(coords["xmin"])
        xmax = "xmax: " + str(coords["xmax"])
        ymin = "ymin: " + str(coords["ymin"])
        ymax = "ymax: " + str(coords["ymax"])
        string = (" " * tab + xmin + "\n" + " " * tab + xmax + "\n" +
                  " " * tab + ymin + "\n" + " " * tab + ymax)
        return string

    def __player_possession(self, pos_lst):
        """
        Input:
            pos_lst [list]: list of ball possession tuples
        Output:
            player_pos {dict}: Returns a dictionary with the player id as the
                                key and a list of tuples of the form
                                (start_frame, end_frame) as the value
        """
        player_pos = {}
        for pos in pos_lst:
            if pos[0] not in player_pos:
                player_pos[pos[0]] = [(pos[1], pos[2])]
            else:
                player_pos[pos[0]].append((pos[1], pos[2]))
        return player_pos

    def __repr__(self) -> str:
        return ("Rim coordinates: \n" +
                self.__print_coords(self.rim, tab=2) + "\n" +
                "Backboard coordinates: \n" +
                self.__print_coords(self.backboard, tab=2) + "\n" +
                "Court lines coordinates: \n" +
                "NONE" + "\n" +
                "Number of frames: " + str(len(self.states)) + "\n" +
                "Number of players: " + str(len(self.players)) + "\n" +
                "Number of passes: " + str(len(self.passes)) + "\n" +
                "Team 1: " + str(self.team1) + "\n" +
                "Team 2: " + str(self.team2) + "\n" +
                "Team 1 Score: " + str(self.score_1) + "\n" +
                "Team 2 Score: " + str(self.score_2))
