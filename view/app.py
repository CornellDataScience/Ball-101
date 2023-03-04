"""
Frontend
"""
import io
import sys
import os
import streamlit as st
import hydralit_components as hc
import pandas as pd
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import main

# Set up tab title and favicon
st.set_page_config(page_title='HoopTracker', page_icon=':basketball:')

# Initialize Session State
# 0 : Default State : No Video Uploaded --> Prompts For Upload / Home Screen Demo
# 1 : Uploading/Processing State --> Loading Screen
# 2 : Done Processing, Show Statistics, Allow Exporting
if 'state' not in st.session_state:
    st.session_state.state = 0
    st.session_state.logo = 'view/media/basketball.png'
    st.session_state.video_file = io.BytesIO(open('view/media/demo_basketball.mov', 'rb').read())
    st.session_state.results = pd.read_csv('view/media/demo_results.csv')
    st.session_state.is_downloaded = False
    st.session_state.upload_name = None


# Backend Connection -----------------------------------------
SERVER_URL = "http://127.0.0.1:8000/"
# Send request to Google Compute Machine
def process_video(video_file):
    response = requests.post(SERVER_URL+"upload", 
                             files={"video_file": video_file}, timeout=30)
    if response.status_code == 200:
        data = response.json()
        st.session_state.upload_name = data.get('message')
    else:
        print('error uploading file') # maybe make an error handler in frontend
    st.session_state.is_downloaded = False
    return video_file is not None


def fetch_csv():
    if not st.session_state.is_downloaded:
        out = requests.get(SERVER_URL+"results", timeout=30)
        st.session_state.results = pd.read_csv(io.StringIO(out.content.decode()))
    st.session_state.is_downloaded = True
    return st.session_state.results.to_csv()

def fetch_processed_video():
    return main.pull_user_video(st.session_state.upload_name)

# ------------------------------------------------------------

# Main Page
def main_page():
    st.markdown('''
        # HoopTracker
        A basketball analytics tracker built on YOLOv5 and OpenCV.
        Simply upload a video in the side bar and click "Process Video."
    ''')

    # send to tips page
    st.button(label="Having Trouble?", on_click=change_state, args=(-1,))

    # Basketball Icon Filler
    _, col2, _ = st.columns([0.5,5,0.5])
    with col2:
        st.image(image=st.session_state.logo,use_column_width=True)


# Tips Page
def tips_page():
    st.markdown('''
        # Tips and Tricks
        Is our model having trouble processing your video? Here's some tips and tricks we have for you.
        * Position the camera on a stable surface, such as a tripod.
        * Ensure the players and hoop and the court are visible as much as possible.
        * We recommend at least 1080p video shot at 30 fps.
    ''')

    # Back to Home
    st.button(label='Back to Home', on_click=change_state, args=(0,))

# Loading Screen
def processing_page():
    st.markdown('''
        # Processing...
        Please wait while we upload and process your video. 
    ''')

    # Loading bar until video processes
    with hc.HyLoader('', hc.Loaders.pulse_bars,):
        process_video(video_file=st.session_state.video_file)

    # Load results page when done
    change_state(2)
    st.experimental_rerun()

# Display Data
def results_page():
    st.video(open(fetch_processed_video(), 'rb').read())

    st.markdown('''
        # Results
        Here are the stats.
    ''')

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(
            data=st.session_state.results,
            x='TEAM',
            y=('REB')
        )

    with col2:
        st.bar_chart(
            data=st.session_state.results,
            x='TEAM',
            y=('TO')
        )


    st.bar_chart(
        data=st.session_state.results,
        x='PLAYER',
        y=('2PA','3PA')
    )

    st.markdown('### Raw Data')
    st.write(st.session_state.results)
    st.download_button(
        label='Download CSV',
        data=fetch_csv(),
        file_name="results.csv")

# Error Page
def error_page():
    st.markdown('''
        # Error: Webpage Not Found
        Try reloading the page to fix the error.
        ''')
    st.button(label='Back to Home', on_click=change_state, args=(0,))

def setup_sidebar():
    # Display upload file widget
    st.sidebar.markdown('# Upload')
    file_uploader = st.sidebar.file_uploader(
        label='Upload a video',
        type=['mp4', 'mov', 'wmv', 'avi', 'flv', 'mkv'],
        label_visibility='collapsed')
    if file_uploader is not None:
        update_video(file_uploader)

    # Display video they uploaded
    st.sidebar.markdown('# Your video')
    st.sidebar.video(data=st.session_state.video_file)

    # Process options to move to next state
    col1, col2 = st.sidebar.columns([1,17])
    consent_check = col1.checkbox(label=" ", label_visibility='hidden')
    col2.caption('''
        I have read and agree to HoopTracker's
        [terms of services.](https://github.com/CornellDataScience/Ball-101)
    ''')

    st.sidebar.button(label='Upload & Process Video',
                      disabled= not consent_check,
                      use_container_width=True,
                      on_click=change_state, args=(1,),
                      type='primary')

    # Write Export Option when in results page
    if st.session_state.state == 2:
        st.sidebar.markdown('# Export')
        st.sidebar.download_button(
            label='Download CSV',
            use_container_width=True,
            data=fetch_csv(),
            file_name="results.csv")

# Call back function to change page
def change_state(state:int):
    st.session_state.state = state

# Updates video on screen
def update_video(video_file):
    st.session_state.video_file = video_file

# Set home info page test
if st.session_state.state == -1:
    tips_page()
elif st.session_state.state == 0:
    main_page()
elif st.session_state.state == 1:
    processing_page()
elif st.session_state.state == 2:
    results_page()
else:
    error_page()

setup_sidebar()
