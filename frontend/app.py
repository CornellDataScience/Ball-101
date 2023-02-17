import streamlit as st
import pandas as pd
import numpy as np
import time
import random


def process_video(uploaded_file):
    st.sidebar.write('You uploaded:')
    st.sidebar.video(data=uploaded_file)

    split_bar = 50
    progress_bar = st.progress(0, text='Please wait. Uploading video...')
    
    for step in range(split_bar):
        time.sleep(0.01 * (step+1))
        progress_bar.progress(step+1, text='Please wait. Uploading video...')

    for step in range(99-split_bar):
        time.sleep(0.01 * (step+1))
        progress_bar.progress(split_bar+step+1, text='Please wait. Processing video...')

    progress_bar.progress(100, text='Processing is successful! See statistics below.')

    st.session_state.state = 2





# Set up tab title and favicon
st.set_page_config(page_title='Hoop Track', page_icon=':basketball:')

# Initialize Session State
# 0 : Default State : No Video Uploaded --> Prompts For Upload / Home Screen Demo
# 1 : Uploading/Processing State --> Loading Screen
# 2 : Done Processing, Show Statistics, Allow Exporting
if 'state' not in st.session_state:
    st.session_state.state = 0

# Set home info page
if st.session_state.state in [0, 1]:
    st.markdown('''
    # HoopTrack
    A basketball analytics tracker built on YOLOv5 and OpenCV.
    Simply upload a video and the results will appear below.
    ''')

# Set up tool side bar for Uploading, Data Display Settings, and Exporting
st.sidebar.markdown('# Upload')

video_upload = st.sidebar.file_uploader(label='Upload a video', type=['mp4', 'mov', 'avi'])

# If they do upload a new video
if video_upload is not None:
    st.session_state.state = 1 # processing
    process_video(video_upload)

    
if st.session_state.state == 2:
        st.sidebar.markdown('# Data Options')
        st.sidebar.markdown('# Export')





