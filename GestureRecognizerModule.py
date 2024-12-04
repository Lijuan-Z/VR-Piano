import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class gestureDetector():
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='play_gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def is_play(self,image_file_name):
        # image = mp.Image.create_from_file(image_file_name)

        # Convert the image to a MediaPipe Image object
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_file_name)

        # Recognize gestures in the input image.
        recognition_result = self.recognizer.recognize(image)
        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            return top_gesture.category_name == 'play'
        return False


if __name__ == "__main__":
    recognizer = gestureDetector()
    isplay = recognizer.is_play('hand_image.png')
    print(isplay)