import sys
import os
import io
import ast
import streamlit as st
import hydralit_components as hc
import pandas as pd
import requests
# from format import Format

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="HoopTracker", page_icon=":basketball:")
# 'set up tab tile and favicon'

# Initialize Session States:
# 0 = Default State : No Video Uploaded --> Prompts For Upload / Home Screen Demo
# 1 = Uploading/Processing State --> Loading Screen
# 2 = Done Processing, Show Statistics, Allow Exporting
if "state" not in st.session_state:
    st.session_state.state = 0
    st.session_state.logo = "src/view/static/basketball.png"
    with open("data/training_data.mp4", "rb") as file:
        st.session_state.video_file = io.BytesIO(file.read())
    st.session_state.processed_video = None
    st.session_state.result_string = None
    st.session_state.upload_name = None
    st.session_state.user_file = "tmp/user_upload.mp4"

# Backend Connection
SERVER_URL = "http://127.0.0.1:8000/"


def process_video(video_file):
    """
    Takes in a mp4 file at video_file and uploads it to the backend, then stores
    the processed video name into session state
    Temporarily: stores the processed video into tmp/user_upload.mp4
    """

    user_video: str = st.session_state.user_file
    # UPLOAD VIDEO
    if video_file is None:
        return False
    r = requests.post(
        SERVER_URL + "upload", files={"video_file": video_file}, timeout=60
    )
    if r.status_code == 200:
        print("Successfully uploaded file")
        data = r.json()
    
        st.session_state.upload_name = data.get("message")
        print(st.session_state.upload_name)
        print("Successfully printed file")
        with open(user_video, "wb") as f:  # TODO is local write; temp fix
            f.write(video_file.getvalue())
    else:
        print("Error uploading file")  # TODO make an error handler in frontend
        return False
    st.session_state.is_downloaded = False

    # PROCESS VIDEO
    try:
        process_response = requests.post(
    SERVER_URL + "process_vid", params={"file_name": st.session_state.upload_name}, timeout=60
)

        

        # Handle processing response
        if process_response.status_code == 200:
            print("Video processing successful")
            if st.session_state.result_string is None:
                print("Warning: result_string is None")
            elif st.session_state.result_string == "":
                print("Warning: result_string is empty")

            # Assuming the backend returns the result string or path to the processed video
            result_data = process_response.json()
            return True
        
        else:
            print(f"Error processing file: {process_response.text}")
            return False

    except requests.RequestException as e:
        print(f"Error during file processing: {e}")
        return False


# Pages
def main_page():
    """
    Loads main page
    """
    st.markdown(
        """
        # HoopTracker
        A basketball analytics tracker built on YOLOv5 and OpenCV.
        Simply upload a video in the side bar and click "Process Video."
    """
    )
    # send to tips page
    st.button(label="Having Trouble?", on_click=change_state, args=(-1,))

    # Basketball Icon Filler
    _, col2, _ = st.columns([0.5, 5, 0.5])
    with col2:
        st.image(image=st.session_state.logo, use_column_width=True)


def loading_page():
    """
    Loads loading page
    """
    st.markdown(
        """
        # Processing...
        Please wait while we upload and process your video. 
    """
    )

    # Loading bar until video processes
    with hc.HyLoader(
        "",
        hc.Loaders.pulse_bars,
    ):
        finished = process_video(video_file=st.session_state.video_file)
        if finished:
            state = 2
        else:
            state = 0  # TODO make error handling + error popup/page

    # Load results page when done
    change_state(state)
    st.rerun()


def results_page():
    st.markdown("## Results\nThese are the results. Here's the processed video and a minimap of the player positions.")

    # Assuming the processed video is stored in st.session_state.processed_video
    if st.session_state.processed_video:
        st.video(open(st.session_state.processed_video, "rb").read())

    st.markdown("## Statistics")

    # Read and display the contents of the results file
    try:
        with open("tmp/results.txt", "r") as file:
            results_data = file.readlines()

        # Parse the file contents into a table format
        # This part depends on the exact format of your results.txt
        # Here is an example for a simple key-value pair format
        results_table = [line.split(": ") for line in results_data if ": " in line]
        st.table(results_table)

    except FileNotFoundError:
        st.error("Results file not found.")
    st.markdown("## MiniMap")
    try:
        video_file_path = 'tmp/minimap.mp4'
        st.video(video_file_path)
    except Exception as e: 
        st.error("Minimap video not found.")
    st.markdown("## Processed Video")
    try:
        p_video_file_path = 'tmp/processed.mp4'
        st.video(p_video_file_path)
    except Exception as e:
        st.error("Processed video not found.")

    st.download_button(
        label="Download Results",
        use_container_width=True,
        data=st.session_state.result_string if st.session_state.result_string else "",
        file_name="results.txt",
    )

    st.button(label="Back to Home", on_click=change_state, args=(0,), type="primary")



def tips_page():
    """
    Loads tips page
    """
    st.markdown(
        """
        # Tips and Tricks
        Is our model having trouble processing your video? Here's some tips and tricks we have for you.
        * Position the camera on a stable surface, such as a tripod.
        * Ensure the players and hoop and the court are visible as much as possible.
        * We recommend at least 1080p video shot at 30 fps.
    """
    )

    # Back to Home
    st.button(label="Back to Home", on_click=change_state, args=(0,))

#Get the results from tmp/results.txt
def get_res():
    file_path = "tmp/results.txt"
    try:
        with open(file_path, "r") as file:
            result_string = file.read()
        result_dict = ast.literal_eval(result_string)
        return result_dict
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

    

def error_page():
    """
    Loads error page
    """
    st.markdown(
        """
        # Error: Webpage Not Found
        Try reloading the page to fix the error.
        If there are any additional issues, please report errors to us 
        [here](https://github.com/CornellDataScience/Ball-101).
        """
    )
    st.button(label="Back to Home", on_click=change_state, args=(0,))


def setup_sidebar():
    """Sets up sidebar for uploading videos"""
    # Display upload file widget
    st.sidebar.markdown("# Upload")
    file_uploader = st.sidebar.file_uploader(
        label="Upload a video",
        type=["mp4", "mov", "wmv", "avi", "flv", "mkv"],
        label_visibility="collapsed",
    )
    if file_uploader is not None:
        update_video(file_uploader)

    # Display video they uploaded
    st.sidebar.markdown("# Your video")
    st.sidebar.video(data=st.session_state.video_file)

    # Process options to move to next state
    col1, col2 = st.sidebar.columns([1, 17])
    consent_check = col1.checkbox(label=" ", label_visibility="hidden")
    col2.caption(
        """
        I have read and agree to HoopTracker's
        [terms of services.](https://github.com/CornellDataScience/Ball-101)
    """
    )

    st.sidebar.button(
        label="Upload & Process Video",
        disabled=not consent_check,
        use_container_width=True,
        on_click=change_state,
        args=(1,),
        type="primary",
    )


# Helpers
def change_state(state: int):
    """
    Call back function to change page
    @param state, integer to change the state
    """
    st.session_state.state = state


def update_video(video_file):
    """
    Updates video on screen
    @param video_file, file path to video
    """
    st.session_state.video_file = video_file


def process_results():
    """
    Processes st.session_state.result_string to display results
    """
    # TODO revamp results processing
    # # Parse the result string into a dictionary
    # result_dict = ast.literal_eval(st.session_state.result_string)

    # # Create dataframes for each category
    # general_df = pd.DataFrame(
    #     {
    #         "Number of frames": [result_dict["Number of frames"]],
    #         "Number of players": [result_dict["Number of players"]],
    #         "Number of passes": [result_dict["Number of passes"]],
    #     }
    # )

    # team_df = pd.DataFrame(
    #     {
    #         "Team 1": [
    #             result_dict["Team 1"],
    #             result_dict["Team 1 Score"],
    #             result_dict["Team 1 Possession"],
    #         ],
    #         "Team 2": [
    #             result_dict["Team 2"],
    #             result_dict["Team 2 Score"],
    #             result_dict["Team 2 Possession"],
    #         ],
    #     },
    #     index=["Players", "Score", "Possession"],
    # )

    # coordinates_df = pd.DataFrame(
    #     {
    #         "Rim coordinates": [result_dict["Rim coordinates"]],
    #         "Backboard coordinates": [result_dict["Backboard coordinates"]],
    #         "Court lines coordinates": [result_dict["Court lines coordinates"]],
    #     }
    # )

    # # Create a dictionary for players data
    # players_dict = {
    #     key: value
    #     for key, value in result_dict.items()
    #     if key not in general_df.columns
    #     and key not in team_df.columns
    #     and key not in coordinates_df.columns
    # }
    # # Remove team score and possession details from players dictionary
    # team_keys = [
    #     "Team 1 Score",
    #     "Team 2 Score",
    #     "Team 1 Possession",
    #     "Team 2 Possession",
    # ]
    # players_dict = {
    #     key: value for key, value in players_dict.items() if key not in team_keys
    # }

    # # Convert player statistics from string to dictionary
    # for player, stats in players_dict.items():
    #     players_dict[player] = ast.literal_eval(stats)

    # players_df = pd.DataFrame(players_dict).T

    # # Display dataframes
    # st.write("### General")
    # st.dataframe(general_df)

    # st.write("### Team")
    # st.dataframe(team_df)

    # st.write("### Players")
    # st.dataframe(players_df)

    # st.write("### Coordinates")
    # st.dataframe(coordinates_df)


# Entry Point
if st.session_state.state == -1:
    tips_page()
elif st.session_state.state == 0:
    main_page()
elif st.session_state.state == 1:
    loading_page()
elif st.session_state.state == 2:
    results_page()
else:
    error_page()

setup_sidebar()
