"""
Backend module built in FastAPI
"""
import time
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse
import pandas as pd
import boto3
import os
import subprocess
from dotenv import load_dotenv
from ..processing.format import Format

from pydantic import BaseModel

class ProcessVideoRequest(BaseModel):
    file_name: str

# Amazon S3 Connection
load_dotenv()
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key)

app = FastAPI()


# Root
@app.get("/")
async def root():
    return {"message": "welcome to hooptracker backend"}


@app.post("/upload")
async def upload_file(video_file: UploadFile = File(...)):
    print("here-upload")
    """
    Upload video file to S3 bucket
    """
    file_name = time.strftime("%Y%m%d-%H%M%S")
    print(file_name)
    print(video_file)
    
    try:
        print("enter try")
        s3.upload_fileobj(video_file.file, "ball-uploads", file_name)
        print("successful upload")
        print(file_name)
        return {"message": file_name, "status": "success"}
    except Exception as ex:
        # raise HTTPException(status_code=500, detail=str(ex))
        return {"error": str(ex)}



@app.post("/process_vid")
async def process_file(file_name: str): # Accept file name as input
 # This will print the raw JSON data sent to the endpoint
    """
    Download video from S3, process it, and optionally upload back to S3
    """
    print("here")
    print(file_name) 
    local_file_name = "temp_video_file"  # Temporary local file name
    processed_file_name = f"processed_{file_name}"  # Name for the processed file

    try:
        print("here")
        # Download the file from S3
        s3.download_file('ball-uploads', file_name, local_file_name)
        print("here")
        # Process the file
        command = ["python", "src/main.py", "--video_file", local_file_name]
        subprocess.run(command)
        print("here")
        # Upload processed file back to S3
        with open(processed_file_name, "rb") as processed_file:
            print("here")
            s3.upload_fileobj(processed_file, "processed-uploads", processed_file_name)
        print("here")
        # Clean up local files
        os.remove(local_file_name)
        os.remove(processed_file_name)

        return {"message": f"Successfully processed {file_name}", "status_code": 200}

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
        print("EROROROROR")
        return {"error": str(ex),"status_code":400}


@app.post("/process")
async def process_file(video_file: UploadFile = File(...)):
    """
    runs main to process file
    TODO change from local to cloud
    """
    try:
        print(video_file.file)
        command = ["python", "src/main.py", "--video_file", video_file]
        subprocess.run(command)
        print("here111")
        return {"message": f"successfully processed {video_file}", "status_code":400}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
        print("EROROROROR")
        return {"error": str(ex),"status_code":400}


# TODO Not being used RN
@app.get("/download/{file_name}")
async def download_file(file_name: str, download_path: str):
    """
    Download video with file_name to download_path
    """
    try:
        s3.download_file("ball-uploads", file_name, download_path)
        return {
            "message": f"successfully downloaded {file_name} to {download_path}",
            "status": "success",
        }
    except Exception as ex:
        return {"error": str(ex)}



@app.get("/results")
async def get_formatted_results():
    try:
        # Assuming the Format.results() method returns the formatted data as JSON
        formatted_data = Format.results()
        return formatted_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/video")
async def get_videos():
    file_path = 'tmp/minimap.mp4'
    def iterfile():  
        with open(file_path, mode="rb") as file_like:  
            yield from file_like 

    return StreamingResponse(iterfile(), media_type="video/mp4")