import cv2
from ultralytics import YOLO


def process_video(source_mov, yolo_weights, classes):
    model = YOLO(yolo_weights)

    cap = cv2.VideoCapture(source_mov)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out_path = source_mov.replace('.mp4', '_processed.mp4')
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (width, height))

    output_data = []
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run YOLO tracking on the frame
            results = model.track(frame, persist=True,
                                  classes=classes, tracker="botsort.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the processed frame to the output video
            out.write(annotated_frame)

            # Save bounding box data
            for res in results:
                output_data.append([(box.xyx2, box.classname)
                                   for box in res.pred])
        else:
            break

    cap.release()
    out.release()

    return output_data, out_path


def get_data(source_mov: str):
    WEIGHTS_PERSON = 'yolov8n.pt'
    WEIGHTS_BASKETBALL = 'yolov8n.pt'  # 'best_basketball.pt'

    person_data, vid_path_person = process_video(
        source_mov, WEIGHTS_PERSON, [1, 2])
    basketball_data, vid_path_basketball = process_video(
        source_mov, WEIGHTS_BASKETBALL, None)

    return {
        'basketball_data': (basketball_data, vid_path_basketball),
        'person_data': (person_data, vid_path_person)
    }


def yolo():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the video file
    video_path = "training_data.mp4"
    cap = cv2.VideoCapture(video_path)
    i = 0
    # Loop through the video frames
    while cap.isOpened():
        print(i)
        i += 1
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            # change tracker to botsort.yaml or bytetrack.yaml
            results = model.track(frame, persist=True,
                                  tracker="botsort.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            # cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    # cv2.destroyAllWindows()


yolo()
