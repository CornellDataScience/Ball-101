"""
Runner module for statistics logic
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stats.general_stats import GeneralStatsProcessor
from stats.team_stats import TeamStatsProcessor
from src.state import StatState

class StatRunner:
    """
    Runner class taking in the video file path and running two passes of
    processing in sequence. 
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.stat_state = StatState()


    def run_general_stats(self):
        """
        Passes video to a general stat processor. Updates stat_state.
        """
        general_processor = GeneralStatsProcessor(self.video_path, self.stat_state)
        general_processor.detect_teams()
        general_processor.compute_possession()
        general_processor.track_ball()


    def run_team_stats(self):
        """
        Passes video to a team stat processor. Updates stat_state.
        """
        team_processor = TeamStatsProcessor(self.video_path, self.stat_state)
        team_processor.compute_team_score()
        team_processor.compute_assists()


    def run(self):
        self.run_general_stats()
        self.run_team_stats()
