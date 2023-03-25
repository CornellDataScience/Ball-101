"""
Runner module for statistics logic
"""
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stats import statistics

def load_config(path):
	with open(path, 'r') as file:
		config = yaml.safe_load(file)
	return config

def run_statistics(video_path):
    statistics.load_video(video_path)
