"""
First pass processing
"""
#import cv2
from src.state import StatState

class GeneralStatsProcessor:
    """
    Class for computing all team-independent statistics. Currently supports:
    team detection, individual possession, ball tracking.
    """
    def __init__(self, video_path, stat_state: StatState):
        self.video_path = video_path
        self.stat_state = stat_state


    def detect_teams(self):
        # Team detection, update self.stat_state
        pass


    def compute_possession(self):
        # Possession calculation, update self.stat_state
        pass

    def track_ball(self):
        # Populate self.stat_state.ball_state
        pass
