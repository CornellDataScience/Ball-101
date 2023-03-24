"""
Second pass processing
"""
#import cv2
from src.state import StatState

class TeamStatsProcessor:
    """
    Class for computing all team-dependent statistics. Currently supports:
    team score, assists.
    """
    def __init__(self, video_path, stat_state: StatState):
        self.video_path = video_path
        self.stat_state = stat_state


    def compute_team_score(self):
        pass

    
    def compute_assists(self):
        pass
