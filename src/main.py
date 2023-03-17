"""
Control Loop Module
"""
import os
import sys
import argparse
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api import gcs
from src import modelrunner, statrunner


def load_config(path):
    """
    Loads the config yaml file to read in parameters and settings.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def pull_user_video(file_name):
    """
    Pulls video from GCS bucket.
    """
    config = load_config('src/config.yaml')
    vars_ = config['vars']
    download_path = vars_['download_path']
    gcs.download_from_bucket(file_name, download_path)
    return download_path


def main(file_name):
    """
    Main control loop. Currently returns the path of the processed video.
    """
    download_path = pull_user_video(file_name)
    processed_path = modelrunner.run_models(download_path)
    statrunner.run_statistics(processed_path)
    return processed_path


def _ci_main(file_path):
    processed_path = modelrunner.run_models(file_path)
    statrunner.run_statistics(processed_path)
    print(f'Model run complete. The output can be found at: {processed_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='HoopTracker',
                  description='feed file into the model')
    parser.add_argument('--ci', metavar='file_path', type=str, required=True,
             help='path of the file to run processing through')
    args = parser.parse_args()
    _ci_main(args.ci)
