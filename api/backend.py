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


@app.post("/upload")
async def upload_file(video_file: UploadFile = File(...)):
    """
    Upload video file to GCS bucket
    """
    blob_name = time.strftime("%Y%m%d-%H%M%S")
    gcs.upload_file_to_bucket(blob_name, video_file.file)
    return {"message": blob_name, "status": "success"}


@app.get("/results")
async def get_results():
    """
    Fetch the results csv from statistics
    """
    dummy_df = pd.read_csv('dummy.csv') # replace with stat logic
    stream = io.StringIO()
    dummy_df.to_csv(stream, index=False)
    response = StreamingResponse(io.StringIO(dummy_df.to_csv(index=False)), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response
