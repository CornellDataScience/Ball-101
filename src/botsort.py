import cv2
from ultralytics import YOLO
from pathlib import Path
import pickle

def get_data_yolov8(source_mov: str, model_path: str):
    """ Returns tracking info using YOLOv8.

    Args:
        source_mov (str): Path to video file.
        model_path (str): Path to the YOLOv8 weights/model.

    Returns:
        dict: Contains 'basketball_data' and 'person_data'.
    """
#testcomment
    # Load the YOLOv8 model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source_mov)
    basketball_data = []
    person_data = []

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, tracker="botsort.yaml")
            print(type(results))
            print(results)
            for detection in results.pred[0]:
                label = detection[4].item()
                # Assuming class index 1 is 'person' and class index 2 is 'basketball'
                if label == 1:
                    person_data.append(detection[:4].tolist())
                elif label == 2:
                    basketball_data.append(detection[:4].tolist())
        else:
            break

    cap.release()

    return {'basketball_data': (basketball_data, source_mov), 'person_data': (person_data, source_mov)}


def sort():
    # TODO: Update with actual paths
    video_path = '/data/training_data.mp4'
    model_path = 'yolov8n.pt'
    
    output = get_data_yolov8(video_path, model_path)
    print("b")
    print(output)
    print(output['basketball_data'])
    return output
    with open('../../tmp/test_output_yolov8.pickle', 'wb') as f:
        pickle.dump(output, f)



'''import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "training_data.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # change tracker to botsort.yaml or bytetrack.yaml
        results = model.track(frame, persist=True, tracker="botsort.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
'''