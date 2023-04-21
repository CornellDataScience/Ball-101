# View
A streamlit-ed front end where users can upload videos of basketball, view, and download results.

## How to run
### Installing dependencies:
```
pip install -r requirements.txt
```

### Running streamlit:
```
streamlit run view/app.py
```

## How to use
### Uploading:
![screenshot of home page](https://github.com/CornellDataScience/Ball-101/blob/clean-infra/view/media/home.png)
1. Upload a valid video file on the sidebar at the top left.
2. Your video will load below it.
3. Click "Process Video."
4. You will be taken to a loading screen as we process the video.

### Exporting:
![screenshot of results page](https://github.com/CornellDataScience/Ball-101/blob/clean-infra/view/media/results.png)
1. After processing, the results page will automatically appear.
2. The results page provides summaries of the player and team data.
3. At the bottom, you adn view and sort the raw data.
4. In the sidebar, export the raw data as a `.csv` file.
