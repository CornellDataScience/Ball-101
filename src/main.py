"""
Control Loop Module
"""
import os
import sys
import yaml
import argparse
from api import gcs
from . import modelrunner
# import statrunner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_config(path):
	with open(path, 'r') as f:
		config = yaml.safe_load(f)
	return config


def pull_user_video(file_name):
	"""
	Pull video from GCS bucket and feed it to ML model
	Download path currently defined to be "model/temp/user.mp4"
	Currently overrides each previous user request.
	"""
	user_file_path = 'model/temp/user.mp4'
	gcs.download_from_bucket(file_name, user_file_path)
	return modelrunner.run_models(user_file_path)


def main():
	config = load_config('config.yaml')
	parser = argparse.ArgumentParser(prog='HoopTracker', 
				  description='feed file into the model')
	parser.add_argument('--ci', metavar='file_name', type=str, required=True,
		     help='path of the file to run processing through')
	

if __name__ == "__main__":
	main()