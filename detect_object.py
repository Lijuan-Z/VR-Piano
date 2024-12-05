import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Visualization constants
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


# Visualization function
def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.

    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualized.

    Returns:
        Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


def run(model: str, camera_id: int, width: int, height: int) -> bool:
    """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the TFLite object detection model.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.

    Returns:
        True if bottle is detected, otherwise False.
    """
    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detection_result_list = []
    bottle_detected = False  # Flag to track bottle detection

    def visualize_callback(result: vision.ObjectDetectorResult,
                           output_image: mp.Image, timestamp_ms: int):
        """Callback to process detection results."""
        nonlocal bottle_detected  # Access the outer function variable

        for detection in result.detections:
            category_name = detection.categories[0].category_name
            score = detection.categories[0].score
            bbox = detection.bounding_box
            print(f'Detected: {category_name} ({score:.2f}), '
                  f'Box: [{bbox.origin_x}, {bbox.origin_y}, {bbox.width}, {bbox.height}]')

            if category_name.lower() == "bottle" and score > 0.5:
                print("Bottle detected!")
                bottle_detected = True  # Set the flag to True if bottle is detected
                return  # Stop processing further detections

        detection_result_list.append(result)

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           running_mode=vision.RunningMode.LIVE_STREAM,
                                           score_threshold=0.5,
                                           result_callback=visualize_callback)
    detector = vision.ObjectDetector.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection using the model.
        detector.detect_async(mp_image, counter)
        current_frame = mp_image.numpy_view()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        if detection_result_list:
            vis_image = visualize(current_frame, detection_result_list[0])
            cv2.imshow('object_detector', vis_image)
            detection_result_list.clear()
        else:
            cv2.imshow('object_detector', current_frame)

        # If bottle is detected, return True and stop
        if bottle_detected:
            print("Bottle Detected, stopping.")
            cap.release()
            cv2.destroyAllWindows()
            return True

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    return False  # Return False if no bottle is detected
