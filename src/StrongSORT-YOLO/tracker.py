from pathlib import Path
import track_v5
#import track_v7
import pickle
import cv2
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

def get_data(source_mov:str):
    """ returns dict as: {
        'basketball_data': (basketball_bounding_boxes, video_path),
        'person_data': (persons_and_rin_bounding_boxes, vide_path)
        }.

    Args:
        source_mov (string path): path to video file
    """
    out_array_pr, vid_path = track_v5.run(source = source_mov, classes= [1, 2],  yolo_weights = 
                                          WEIGHTS / 'best.pt', save_vid=False, ret=True)
    out_array_bb, bb_vid_path = track_v5.run(source = source_mov, 
                                             yolo_weights = WEIGHTS / 'best_basketball.pt', 
                                             save_vid=False, ret=True, skip_big=True)
    print(out_array_pr)
    return {'basketball_data': (out_array_bb, bb_vid_path), 'person_data': (out_array_pr, vid_path)}

import cv2
from ultralytics import YOLO

def get_data_yolov8(source_mov: str):
    """Returns a dictionary containing bounding box data and video paths for basketballs and persons/rings.

    Args:
        source_mov (str): Path to the video file.
    
    Returns:
        dict: A dictionary with basketball and person data.
    """
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the video file
    cap = cv2.VideoCapture(source_mov)

    basketball_bboxes = []
    persons_and_rings_bboxes = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            print(results)
            # Assume detections can be accessed from results.detections (this may need modification)
            for det in results.detections:  # This line may need modification
                bbox = det.tolist()  # Assuming this gives bounding box data
                # Classify the detections (you'll need to replace this with actual classification logic)
                if det.class_label == 'basketball':  # This line may need modification
                    basketball_bboxes.append(bbox)
                else:
                    persons_and_rings_bboxes.append(bbox)
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    return {
        'basketball_data': (basketball_bboxes, source_mov),
        'person_data': (persons_and_rings_bboxes, source_mov)
    }

# Call the function with the path to your video file
#output = get_data_yolov8('path/to/video.mp4')



def test():
    print('import worked')

if __name__ == '__main__':
    # TODO change to actual video path
    #output = get_data_yolov8('../../data/training_data.mp4')
    output = get_data('../../data/training_data.mp4')
    print(output['basketball_data'])
    with open('../../tmp/test_output.pickle', 'wb') as f:
        pickle.dump(output, f)
    
