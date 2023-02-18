# Ball-101
Building a service for the community at large + low budget sports programs for Sports Analytics and Stats Tracking

## Project Structure

```bash
├── data
│   ├── gcs
│   │   ├── gcs.py
│   │   │── credentials.json
│   ├── local-data
│   ├── models-data
│   ├── parser.py
├── src
│   ├── models
│   ├── stats
│   ├── view
│   ├── config.py
│   ├── main.py
│   ├── state.py
├── test
```

## Data 
### GCS
Contains all Google Cloud requests and actions, hosts authentication, handles data loading and unloading (ETL) to Google Cloud Storage

### local-data
Holds video file parsed through GCS (pkl format transformed into mp4 by GCS)

### models-data
Holds data for trained models

### parser.py
Performs data processing of the video file to be fed into models


## src
### models
Holds the ML models that use YOLOv5/OpenCV for object detection and tracking. Takes in preprocessed data from parser

### stats
Logic for statistics calculations as the video is parsed and tracked. Currently contains one module (statistics.py) but may be expanded in the future to incorporate more detailed analytics.

### view
frontend of the app, built in streamlit

### state.py
holds the object state of the game being tracked (combine with stats package?)
