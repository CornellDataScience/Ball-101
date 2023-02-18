import os
from google.cloud import storage

# GCS client setup
client = storage.Client.from_servie_account_json('credentials.json')
#bucket = client.get_bucket() #bucket name

# upload
# blob = bucket.blob('')
# blob.upload_from_filename('')

# download
# blob = bucket.blob('')
# blob.download_to_filename('')