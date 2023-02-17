import streamlit as st
import hydralit_components as hc
import pandas as pd
import numpy as np
import time
from io import BytesIO


# Initialize Session State
# 0 : Default State : No Video Uploaded --> Prompts For Upload / Home Screen Demo
# 1 : Uploading/Processing State --> Loading Screen
# 2 : Done Processing, Show Statistics, Allow Exporting
if 'state' not in st.session_state:
    st.session_state.state = 0


# Set up tab title and favicon
st.set_page_config(page_title='Hoop Track', page_icon=':basketball:')


# Main Page
def main_page():
    st.markdown('''
        # HoopTrack
        A basketball analytics tracker built on YOLOv5 and OpenCV.
        Simply upload a video in the side bar, click "Process Video," and the results will appear below.
    ''')

    # send to tips page
    tips_button = st.button(label="Having Trouble?", on_click=change_state, args=(-1,))

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
    back_button = st.button(label='Back to Home', on_click=change_state, args=(0,))

# Loading Screen
def processing_page():
    st.markdown('''
        # Processing...
        Please wait while we are processing your video. 
        Your call is very important to us. 
        Thank you for shopping at Aperture Science.
    ''')

    with hc.HyLoader('', hc.Loaders.pulse_bars,):
        time.sleep(5)
    
    st.button(label='Proceed to Results', on_click=change_state, args=(2,))

# Display Data
def results_page():
    st.markdown('''
        # Results
        Here are the results you numb knuckles.
    ''')

# Sidebar
def setup_sidebar():
    # Display upload file widget
    st.sidebar.markdown('# Upload')
    file_uploader = st.sidebar.file_uploader(label='Upload a video', type=['mp4', 'mov', 'avi'])

    # Display video they uploaded
    st.sidebar.write('Your video')
    video_file = 'frontend/demo_basketball.mp4'
    if file_uploader is not None:
        video_file = file_uploader
    st.sidebar.video(data=video_file)

    # Process options to move to next state
    process_button = st.sidebar.button(label='Process Video', use_container_width=True, on_click=change_state, args=(1,))
    if process_button:
        process_video(video_file)



def process_video(video_file):
    #
    return

def change_state(state:int):
    st.session_state.state = state
    print('state changed to ' + str(state))

st.session_state.state

# Set home info page test i
match st.session_state.state:
    case -1: tips_page()
    case 0: main_page()
    case 1: processing_page()
    case 2: results_page()

setup_sidebar()
