import os
from google.cloud import storage

# GCS client setup
path = os.path.dirname(__file__)
credentials = os.path.join(path, 'credentials.json')
storage_client = storage.Client.from_service_account_json(credentials)
bucket = storage_client.get_bucket('app_user_upload')

def change_bucket(bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception as e:
        print(e)

# Upload
def upload_to_bucket(blob_name, file_path):
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
    except Exception as e:
        print(e)

def upload_file_to_bucket(blob_name, file):
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_file(file)
    except Exception as e:
        print(e)

# Download
def download_from_bucket(blob_name, file_path):
    try:
        blob = bucket.blob(blob_name)
        blob.download_to_filename(file_path)
    except Exception as e:
        print(e)
