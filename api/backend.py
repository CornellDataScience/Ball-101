"""
Backend module built in FastAPI
"""
import time
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import gcs

app = FastAPI()

# Root
@app.get("/")
async def root():
    return {"message": "welcome to hooptracker backend"}


# Upload video file to GCS bucket
@app.post("/upload")
async def upload_file(video_file: UploadFile = File(...)):
    blob_name = time.strftime("%H%M%S-%Y%m%d")
    gcs.upload_file_to_bucket(blob_name, video_file.file)
    return {"message": "Upload success"}


# Fetch results CSV
@app.get("/results")
async def get_results():
    dummy_df = pd.read_csv('dummy.csv') # replace with stat logic
    stream = io.StringIO()
    dummy_df.to_csv(stream, index=False)
    response = StreamingResponse(io.StringIO(dummy_df.to_csv(index=False)), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response
