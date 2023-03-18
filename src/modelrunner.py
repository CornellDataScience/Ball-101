"""
Runner module for ML models
"""
import sys
import os
from model.yolov5 import detect
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_models(user_file_path, model_configs):
    """
    Takes the downloaded video from user_file_path and feeds to YOLOV5.
    Returns the path of the processed (and reencoded) video file, currently titled "new.mp4".
    """
    model_vars = model_configs
    export_path = model_vars['export_path']
    name = model_vars['name']
    output_file_name = model_vars['output_file_name']
    file_name = model_vars['file_name']
    frame_reduction_factor = model_vars['frame_reduction_factor']
    drop_frames(f'{export_path}/{output_file_name}', frame_reduction_factor)
    detect.run(source=user_file_path, project=export_path,
               name=name, exist_ok=True)
    reencode(f'{export_path}/{name}/{output_file_name}',
             f'{export_path}/{name}/{file_name}')
    return os.path.join(export_path, name, file_name)


def reencode(input_path, output_path):
    """
    Re-encodes a MPEG4 video file to H.264 format. Overrides existing output videos if present.
    Deletes the unprocessed video when complete.
    """
    reencode_command = f'ffmpeg -y -i {input_path} -vcodec libx264 -c:a copy {output_path}'
    os.system(reencode_command)
    os.remove(input_path)


def drop_frames(input_path, reduction_factor):
    """
    Alters the input video fps to to 1 / reduction_factor.
    """
    video = cv2.VideoCapture(input_path)
    nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = cv2.VideoWriter('model/temp/user2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                                   int(video.get(cv2.CAP_PROP_FPS)/2), (int(video.get(
        cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for i in range(nframes):
        ret, frame = video.read()
        if not ret:
            break
        if i % reduction_factor == 0:
            output_video.write(frame)

    video.release()
    output_video.release()
    os.remove(input_path)
    os.rename('model/temp/user2.mp4', input_path)
