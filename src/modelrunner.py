"""
Runner module for ML models
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.yolov5 import detect


def run_models(user_file_path):
    """
    Takes the downloaded video from user_file_path and feeds to YOLOV5.
    Returns the path of the processed (and reencoded) video file, currently titled "new.mp4".
    """
    export_path = 'model/temp'
    name = 'output'
    file_name = 'new.mp4'
    detect.run(source=user_file_path, project=export_path, name=name, exist_ok=True)
    reencode_command = 'ffmpeg -i model/temp/output/user.mp4 -vcodec libx264 model/temp/output/new.mp4'
    os.system(reencode_command)
    return os.path.join(export_path, name, file_name)
