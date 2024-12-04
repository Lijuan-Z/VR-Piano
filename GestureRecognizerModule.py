import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class gestureDetector():
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='play_gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def is_play(self,image):
        # image = mp.Image.create_from_file(image)
        image = mp.Image(format=mp.ImageFormat.SRGB, data=image)

        # STEP 4: Recognize gestures in the input image.
        recognition_result = self.recognizer.recognize(image)

        top_gesture = recognition_result.gestures[0][0]
        return top_gesture.category_name == 'play'

if __name__ == "__main__":
    recognizer = gestureDetector()
    isplay = recognizer.is_play('hand_image.jpg')
    print(isplay)