"""
Control Loop Module
"""
import os
import sys
import argparse
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api import gcs
from src.modelrunner import ModelRunner
from src.statrunner import StatRunner


def load_config(path):
    """
    Loads the config yaml file to read in parameters and settings.
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def pull_user_video(file_name):
    """
    Pulls video from GCS bucket. Returns the path of the downloaded video.
    """
    config = load_config('src/config.yaml')
    vars_ = config['vars_']
    download_path = vars_['download_path']
    gcs.download_from_bucket(file_name, download_path)
    return download_path


def main(file_name, file_path=None):
    """
    Main control loop. Currently returns the path of the processed video.
    For normal executions, file_name refers to the timestamped name of the uploaded file.
    file_path is only specified when CI testing. 
    """
    config = load_config('src/config.yaml')
    model_vars = config['model_vars']
    stat_vars = config['stat_vars']
    video_path = pull_user_video(file_name) if (file_path is None) else file_path
    modelrunner = ModelRunner(video_path, model_vars)
    modelrunner.run()
    processed_path = modelrunner.get_processed_video()
    text_output_path = modelrunner.get_text_output()
    statrunner = StatRunner(processed_path, text_output_path, stat_vars)
    statrunner.run()
    print(f'Model run complete. The output can be found at: {processed_path}')
    return processed_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='HoopTracker',
                  description='feed file into the model')
    parser.add_argument('--ci', metavar='file_path', type=str, required=True,
             help='path of the file to run processing through')
    args = parser.parse_args()
    main(None, file_path=args.ci)
