import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import gcs

# Test upload
gcs.upload_to_bucket('test_blob', '../data/local-data/demo_results.csv')