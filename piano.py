import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm
import concurrent.futures
from playsound import playsound

soundPool = concurrent.futures.ThreadPoolExecutor(max_workers=20)

def play_sound(key):
    match key:
        case 'A':
            playsound("./sound/A_major.wav")
        case 'B':
            playsound("./sound/B_major.wav")
        case 'C':
            playsound("./sound/C_major.wav")
        case 'D':
            playsound("./sound/D_major.wav")
        case 'E':
            playsound("./sound/D_major.wav")


CVID = 0
CV_DELAY = 1
wCam, hCam = 1024, 768

# FPS
pTime = 0
FPS_X_LOCATION, FPS_Y_LOCATION = 30, 40
FPS_FONT = cv2.FONT_HERSHEY_COMPLEX
FPS_FONT_SCALE, FPS_FONT_THINKNESS = 1, 2
FPS_FONT_COLOR = (255, 0, 0)

# Finger drawing
FINGER_DOT_SIZE = 10
FINGER_DOT_COLOR = (255, 0, 255)
FINGER_DOT_SHAPE = cv2.FILLED
LINE_FROM_FINGER_COLOR = (0, 255, 255)
LINE_FROM_FINGER_THICKNESS = 2

# Touching point (hand)
TOUCHING_POINT_COLOR = (255, 255, 255)
TOUCHING_POINT_SHAPE = cv2.FILLED
TOUCHING_POINT_SIZE = 15

# Touching point (piano bar)
TOUCHING_ZONE_FROM_PIANO_BAR_CENTER = 70
READY_ZONE = 150
BAR_DOWN_RANGE = (15, 70)

PIANO_BAR_COLOR = (0, 255, 0)
PIANO_BARS_CENTER_POS = (325, 425)
BAR_SIZE_FROM_CENTER = 20       # KEY FACTOR TO CONTROL THE SIZE OF PIANO BARS AND ALSO THE AREA OF TOUCH
PIANO_BAR_A_CENTER = (PIANO_BARS_CENTER_POS[0] - 4* BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'A')
PIANO_BAR_B_CENTER = (PIANO_BARS_CENTER_POS[0] - 2* BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'B')
PIANO_BAR_C_CENTER = (PIANO_BARS_CENTER_POS[0] + 0, PIANO_BARS_CENTER_POS[1], 'C')
PIANO_BAR_D_CENTER = (PIANO_BARS_CENTER_POS[0] + 2* BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'D')
PIANO_BAR_E_CENTER = (PIANO_BARS_CENTER_POS[0] + 4* BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'E')

PIANO_BARS = [PIANO_BAR_A_CENTER, PIANO_BAR_B_CENTER, PIANO_BAR_C_CENTER, PIANO_BAR_D_CENTER, PIANO_BAR_E_CENTER]
FINGERTIPS = [8] # currently one finger
# FINGERTIPS = [4, 8, 12, 16, 20]

THROTTLE_THRESHOLD = 100     # if a finger stay at the same bar. This control how many conservative bars to wait before we allow another sound playing

cap = cv2.VideoCapture(CVID)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(min_detection_confidence=0.8)


bar_label = ''      # a global variable to know which piano bar it is touching
trottle_control = []    # a global variable to store conservative bars. It use to prevent keep firing the same key when a finger stay in touching position


def get_positions_by_bar_name(bar):
    match bar:
        case 'A':
            return (PIANO_BAR_A_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_A_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_A_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_A_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'B':
            return (PIANO_BAR_B_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_B_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_B_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_B_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'C':
            return (PIANO_BAR_C_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_C_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_C_CENTER[0] + BAR_SIZE_FROM_CENTER,PIANO_BAR_C_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'D':
            return (PIANO_BAR_D_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_D_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_D_CENTER[0] + BAR_SIZE_FROM_CENTER,PIANO_BAR_D_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'E':
            return (PIANO_BAR_E_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_E_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_E_CENTER[0] + BAR_SIZE_FROM_CENTER,PIANO_BAR_E_CENTER[1] + BAR_SIZE_FROM_CENTER)


def piano_bar(img, bar, pos, bar_label):
    # Drawing current piano bar with real-time status
    # print(bar, bar_label)
    x1, y1, x2, y2 = get_positions_by_bar_name(bar)

    if bar == bar_label:
        # bar down
        pos = np.interp(pos, [BAR_DOWN_RANGE[0], BAR_DOWN_RANGE[1]], [y2, y1])
        # print(int(pos))

        cv2.line(img, (x1, int(pos)), (x2, int(pos)), PIANO_BAR_COLOR, 3)  # top, both p1, p2 y move
        cv2.line(img, (x1, y1), (x1, int(pos)), PIANO_BAR_COLOR, 3)  # left,  y2 move
        cv2.line(img, (x2, y1), (x2, int(pos)), PIANO_BAR_COLOR, 3)  # right, y2 move
        cv2.line(img, (x1, y2), (x2, y2), PIANO_BAR_COLOR, 3)  # bottom, never move
        # if int(pos) < y2:
        cv2.rectangle(img, (x1, int(pos)), (x2, y2), PIANO_BAR_COLOR, cv2.FILLED)

    else:
        # No touching, draw normal bar
        cv2.line(img, (x1, y1), (x2, y1), PIANO_BAR_COLOR, 3)  # top, both p1, p2 y move
        cv2.line(img, (x1, y1), (x1, y2), PIANO_BAR_COLOR, 3)  # left,  y2 move
        cv2.line(img, (x2, y1), (x2, y2), PIANO_BAR_COLOR, 3)  # right, y2 move
        cv2.line(img, (x1, y2), (x2, y2), PIANO_BAR_COLOR, 3)  # bottom, never move

def throttle_controller(bar):
    # True: wait enough time, allow to play the sound
    # False: not enough waiting, don't allow any sound play
    global trottle_control

    if len(trottle_control) == 0:
        # the first bar, always allow to play conservatively
        trottle_control.append(bar)
        return True
    elif len(trottle_control) < THROTTLE_THRESHOLD:
        if trottle_control[-1] == bar:
            # only add bar if the current bar is the same as last bar
            trottle_control.append(bar)
            return False
        else:
            # otherwise it is another bar and we should allow and clear the list
            trottle_control = []
            return True
    else:
        # The bar is already pass threshold, allow to play
        trottle_control = []
        return True

def finger_to_keys_distance(img, finger_position_arr):
    # 1. determine the closes key by x first, then y
    # 2. determine if the distance is in ready, but not touch
    # 3. determine if it is touch
    # 4. determine the speed for loudness (extra)
    # finger (x, y)
    # piano bars, each central x, central y
    for bar in PIANO_BARS:

        if finger_position_arr[0] > bar[0] - 25 and finger_position_arr[0] < bar[0] + 25:
            print(f"{bar[2]} is testing range {bar[0] - 25} > {finger_position_arr[0]} < {bar[0] + 25}")
        # if within the bar width, calculate distance, draw line if within ready zone
            distance = math.hypot(finger_position_arr[0] - bar[0], finger_position_arr[1] - bar[1])
            # print(f"in key {bar[2]}, {distance}")
            if distance <= READY_ZONE:
                print(f"READY_ZONE {bar[2]}, {distance}")
                # draw on finger
                cv2.circle(img, (finger_position_arr[0], finger_position_arr[1]), FINGER_DOT_SIZE, FINGER_DOT_COLOR, FINGER_DOT_SHAPE)
                # draw line
                cv2.line(img, (finger_position_arr[0], finger_position_arr[1]), (bar[0], bar[1]), LINE_FROM_FINGER_COLOR, LINE_FROM_FINGER_THICKNESS)
                # draw dots on bar also
                cv2.circle(img, (bar[0], bar[1]), FINGER_DOT_SIZE, FINGER_DOT_COLOR, FINGER_DOT_SHAPE)

                # return the ready distance and also the bar label
                return (distance, bar[2])

    return (math.inf, "")





piano_bar_info = (9999, '') # initialize, info for calling piano_bar function
while True:
    success, img = cap.read() # initialize cv
    img = detector.findHands(img) # self create drawing hands class
    lmList = detector.findPosition(img, draw=False)

    # draw finger pt and lines
    if len(lmList) != 0:
        for finger in FINGERTIPS:
            x2, y2 = lmList[finger][1], lmList[finger][2]
            cv2.circle(img, (x2, y2), FINGER_DOT_SIZE, (125, 125, 0), FINGER_DOT_SHAPE)

            dist, bar_label = finger_to_keys_distance(img, [x2, y2]) # return the distance and draw the line if distance is short enough

            print(f"pause on conservative key: {len(trottle_control)} [0 - ${THROTTLE_THRESHOLD}]")
            if bar_label == "":
                # finger is not on a bar or left the bar, clear the trottle control:
                trottle_control = []

            if dist <= TOUCHING_ZONE_FROM_PIANO_BAR_CENTER:
                # consider touching
                print(f"finger {finger} is touched bar {bar_label} with distance: {dist}")

                # check if it is a conservative keys. Not playing sound if the finger is just keep staying

                if throttle_controller(bar_label):
                    soundPool.submit(play_sound, bar_label)
                piano_bar_info = (dist, bar_label)


    for bar in PIANO_BARS:
        piano_bar(img, bar[2], int(piano_bar_info[0]), bar_label)



    # fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (FPS_X_LOCATION, FPS_Y_LOCATION), FPS_FONT, FPS_FONT_SCALE,
                FPS_FONT_COLOR, FPS_FONT_THINKNESS)

    # show cv
    cv2.imshow("Img", img)
    cv2.waitKey(CV_DELAY)
