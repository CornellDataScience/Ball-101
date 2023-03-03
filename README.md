# Ball-101
Building a service for the community at large + low budget sports programs for Sports Analytics and Stats Tracking

## Project Structure

```bash
├── data
│   ├── gcs # google cloud storage requests and actions
│   │   ├── gcs.py # handles data loading/unloading
│   │   │── credentials.json # gcs authentication
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

<div style="width: 640px; height: 480px; margin: 10px; position: relative;"><iframe allowfullscreen frameborder="0" style="width:640px; height:480px" src="https://lucid.app/documents/embedded/5283e500-286d-4b60-9a31-6ab294866694" id="M3xyY~qxCJ1e"></iframe></div>