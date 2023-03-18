"""
Helper API module for handling Google Cloud Storage Calls
"""
import os
from google.cloud import storage

# GCS client setup
path = os.path.dirname(__file__)
credentials = os.path.join(path, 'credentials.json')
storage_client = storage.Client.from_service_account_json(credentials)
bucket = storage_client.get_bucket('app_user_upload')


def change_bucket(bucket_name):
    """Change the current GCS bucket to upload to/download from."""
    try:
        global bucket
        bucket = storage_client.get_bucket(bucket_name)
    except Exception as e:
        print(e)


def upload_to_bucket(blob_name, file_path):
    """Upload through file path"""
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
    except Exception as e:
        print(e)


def upload_file_to_bucket(blob_name, file):
    """Upload through file name"""
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_file(file)
    except Exception as e:
        print(e)


def download_from_bucket(blob_name, file_path):
    """Download to file path"""
    try:
        blob = bucket.blob(blob_name)
        blob.download_to_filename(file_path)
    except Exception as e:
        print(e)
