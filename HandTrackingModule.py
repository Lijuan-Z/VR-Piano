import cv2
import time
import mediapipe as mp

CVID = 0
CV_DELAY = 1
wCam, hCam = 1024, 768

# FPS
FPS_X_LOCATION, FPS_Y_LOCATION = 30, 40
FPS_FONT = cv2.FONT_HERSHEY_COMPLEX
FPS_FONT_SCALE, FPS_FONT_THINKNESS = 1, 2
FPS_FONT_COLOR = (255, 0, 0)

class handDetector():
    def __init__(self,mode=False, max_num_hands = 2, model_complexity = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.mode = mode
        self.maxHands = max_num_hands
        self.modeComplexity = model_complexity
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence

        # detecting hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,  # Whether to treat input images as a batch of static images or a video stream
            max_num_hands=2,         # Maximum number of hands to detect
            min_detection_confidence=0.5,  # Minimum confidence value to consider detection successful
            min_tracking_confidence=0.5    # Minimum confidence for hand tracking to be considered successful
        )

        self.mpDraw = mp.solutions.drawing_utils  # the 21 dot matrix of the hand

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNum = 1, draw = True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(f"id {id}: {cx}, {cy}")
                lmList.append([id, cx, cy])

            if handNum == 2 and len(self.results.multi_hand_landmarks) > 1:
                myHand2 = self.results.multi_hand_landmarks[1]

                for id, lm in enumerate(myHand2.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id + 21, cx, cy])


                if draw:
                    # test draw a circle at dot id at each finger tip:
                    if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList





def main():
    cap = cv2.VideoCapture(CVID)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0

    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        # if len(lmList) != 0:
        #     print(lmList[4])

        # fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (FPS_X_LOCATION, FPS_Y_LOCATION), FPS_FONT, FPS_FONT_SCALE,
                    FPS_FONT_COLOR, FPS_FONT_THINKNESS)

        # show cv
        cv2.imshow("Img", img)
        cv2.waitKey(CV_DELAY)


if __name__ == "__main__":
    main()
