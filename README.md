# Ball-101
Building a service for the community at large + low budget sports programs for Sports Analytics and Stats Tracking.

## Project Structure

```bash
├── api
│   ├── backend.py # fastapi backend server
│   ├── credentials.json # gcs authentication
│   ├── gcs.py # google cloud storage loading/unloading
├── data
│   ├── local-data # model testing data
│   ├── models-data # trained model data
│   ├── loader.py # (optional) data loader
├── src
│   ├── config.json # global configs for models and stats
│   ├── main.py
│   ├── modelrunner.py
│   ├── statrunner.py
│   ├── state.py # object state of tracked game
│   ├── utils.py
├── models # ml models: yolov5 and opencv
├── stats # calculations logic
│   ├── statistics.py
├── view # app frontend
│   ├── app.py
├── test
```

## Setup Instructions
To get started, clone the repo and install requirements.txt in a Python>=3.8.0 environment.
```
git clone https://github.com/CornellDataScience/Ball-101
cd Ball-101
pip3 install -r requirements.txt
```
Fill in GCS credentials under api/credentials.json.

Start the server backend by running
```
cd api
uvicorn backend:app --reload
```

## LucidChart Pipeline Diagram 
https://lucid.app/documents/embedded/5283e500-286d-4b60-9a31-6ab294866694?invitationId=inv_28b23f10-a3e4-4131-b291-fa378f5a1b4e#