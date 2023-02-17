"""
Parses and processes video into analysable data for models.
"""
import pandas as pd
import cv2

# Get video from data stream
def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    pass

def parse_video(video):
    frames = []
    pass