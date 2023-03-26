"""
Runner module for ML models
"""
import sys
import os
from model.yolov5 import detect
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ModelRunner:
    """
    Class for executing the YOLOV5 model on a specified video path. 
    """
    def __init__(self, video_path, model_configs):
        self.video_path = video_path
        model_vars = model_configs
        self.export_directory = model_vars['export_directory']
        self.export_folder_name = model_vars['export_folder_name']
        self.export_filename = model_vars['export_filename']
        self.reencoded_filename = model_vars['reencoded_filename']
        self.frame_reduction_factor = model_vars['frame_reduction_factor']


    def reencode(self, input_path, output_path):
        """
        Re-encodes a MPEG4 video file to H.264 format. Overrides existing output videos if present.
        Deletes the unprocessed video when complete.
        """
        reencode_command = f'ffmpeg -y -i {input_path} -vcodec libx264 -c:a copy {output_path}'
        os.system(reencode_command)
        os.remove(input_path)


    def drop_frames(self, input_path):
        """
        Alters the input video fps to 1 / reduction_factor.
        """
        dummy_path = 'model/temp/user2.mp4'
        video = cv2.VideoCapture(input_path)
        nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        output_video = cv2.VideoWriter(dummy_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                    int(video.get(cv2.CAP_PROP_FPS)/2), (int(video.get(
            cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        for i in range(nframes):
            ret, frame = video.read()
            if not ret:
                break
            if i % self.frame_reduction_factor == 0:
                output_video.write(frame)

        video.release()
        output_video.release()
        os.remove(input_path)
        os.rename(dummy_path, input_path)


    def run(self):
        """
        Executes YOLOV5 models and its related video pre- and post- processing.
        """
        self.drop_frames(f'{self.export_directory}/{self.export_filename}')
        detect.run(source=self.video_path, project=self.export_directory,
               name=self.export_folder_name, exist_ok=True)
        self.reencode(f'{self.export_directory}/{self.export_folder_name}/{self.export_filename}',
             f'{self.export_directory}/{self.export_folder_name}/{self.reencoded_filename}')


    def get_processed_video(self):
        """Fetches the location of the processed video."""
        return os.path.join(self.export_directory, self.export_folder_name, self.reencoded_filename)

    def get_text_output(self):
        """TODO Returns the location of the model text output."""
        return None
