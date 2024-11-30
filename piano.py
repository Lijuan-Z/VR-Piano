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

# Size of piano bars in square. e.g. 20 = 40 width * 40 height
BAR_SIZE_FROM_CENTER = 20       # KEY FACTOR TO CONTROL THE SIZE OF PIANO BARS

# Touching point (hand)
TOUCHING_POINT_COLOR = (255, 255, 255)
TOUCHING_POINT_SHAPE = cv2.FILLED
TOUCHING_POINT_SIZE = 15

# Touching point (piano bar)
TOUCHING_ZONE_FROM_PIANO_BAR_CENTER = BAR_SIZE_FROM_CENTER + 15  # KEY FACTOR TO DEFINE THE STARTING POINT OF TOUCH IN Y-AXIS
READY_ZONE = 80
BAR_DOWN_RANGE = (15, TOUCHING_ZONE_FROM_PIANO_BAR_CENTER)
TOUCHING_ZONE_COLOR = (0, 0, 255)
TOUCHING_ZONE_WIDTH_DEDUCTION = 10

PIANO_BAR_COLOR = (0, 255, 0)
PIANO_BARS_CENTER_POS = (325, 425)
PIANO_BAR_A_CENTER = (PIANO_BARS_CENTER_POS[0] - 4* BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'A')
PIANO_BAR_B_CENTER = (PIANO_BARS_CENTER_POS[0] - 2* BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'B')
PIANO_BAR_C_CENTER = (PIANO_BARS_CENTER_POS[0] + 0, PIANO_BARS_CENTER_POS[1], 'C')
PIANO_BAR_D_CENTER = (PIANO_BARS_CENTER_POS[0] + 2* BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'D')
PIANO_BAR_E_CENTER = (PIANO_BARS_CENTER_POS[0] + 4* BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'E')

PIANO_BARS = [PIANO_BAR_A_CENTER, PIANO_BAR_B_CENTER, PIANO_BAR_C_CENTER, PIANO_BAR_D_CENTER, PIANO_BAR_E_CENTER]
FINGERTIPS = [8, 12] # currently one finger
# FINGERTIPS = [4, 8, 12, 16, 20]
NUMBER_OF_HANDS = 0 # one or two hands

THROTTLE_THRESHOLD = 100     # if a finger stay at the same bar. This control how many conservative bars to wait before we allow another sound playing

# screen control
SHOW_FINGERTIP_DOTS_AND_LINES = True
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


def draw_single_piano_bar(img, bar, pos):
    # Drawing single piano bar with real-time y-axis of pressed bar
    # print(bar, bar_label)
    x1, y1, x2, y2 = get_positions_by_bar_name(bar)
    if pos is not None:
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

def throttle_controller(trottle_control, finger, bar):
    # True: wait enough time, allow to play the sound
    # False: not enough waiting, don't allow any sound play
    # print(trottle_control)

    if len(trottle_control[finger]) == 0:
        # the first bar, always allow to play conservatively
        trottle_control[finger].append(bar)
        return True
    elif len(trottle_control[finger]) < THROTTLE_THRESHOLD:
        if trottle_control[finger][-1] == bar:
            # only add bar if the current bar is the same as last bar
            trottle_control[finger].append(bar)
            return False
        else:
            # otherwise it is another bar and we should allow and clear the list
            trottle_control[finger] = []
            return True
    else:
        # The bar is already pass threshold, allow to play
        trottle_control[finger] = []
        return True

def finger_to_keys_distance(img, finger_position_arr):
    # 1. determine the closes key by x first, then y
    # 2. determine if the distance is in ready, but not touch
    # 3. determine if it is touch
    # 4. determine the speed for loudness (extra)
    # finger (x, y)
    # piano bars, each central x, central y
    for bar in PIANO_BARS:
        LEFT_TOUCHING_MARGIN = bar[0] - BAR_SIZE_FROM_CENTER + TOUCHING_ZONE_WIDTH_DEDUCTION
        RIGHT_TOUCHING_MARGIN = bar[0] + BAR_SIZE_FROM_CENTER - TOUCHING_ZONE_WIDTH_DEDUCTION
        # Staring by consider only finger position within the range of current bar. The total width of the bar
        if finger_position_arr[0] > LEFT_TOUCHING_MARGIN and finger_position_arr[0] < RIGHT_TOUCHING_MARGIN:
            # print(f"{bar[2]} is testing range {LEFT_TOUCHING_MARGIN} > {finger_position_arr[0]} < {RIGHT_TOUCHING_MARGIN}")
            if SHOW_FINGERTIP_DOTS_AND_LINES:
                # visualize the area of touch
                # overlay = img.copy()
                cv2.rectangle(img, (LEFT_TOUCHING_MARGIN, PIANO_BARS_CENTER_POS[1] - TOUCHING_ZONE_FROM_PIANO_BAR_CENTER), (RIGHT_TOUCHING_MARGIN, PIANO_BARS_CENTER_POS[1]), TOUCHING_ZONE_COLOR, cv2.FILLED)
                # img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # if within the bar width, calculate distance, draw line if within ready zone
            distance = math.hypot(finger_position_arr[0] - bar[0], finger_position_arr[1] - bar[1])
            # print(f"in key {bar[2]}, {distance}")
            if distance <= READY_ZONE:
                # print(f"READY_ZONE {bar[2]}, {distance}")

                if SHOW_FINGERTIP_DOTS_AND_LINES:
                    # draw on finger
                    cv2.circle(img, (finger_position_arr[0], finger_position_arr[1]), FINGER_DOT_SIZE, FINGER_DOT_COLOR, FINGER_DOT_SHAPE)
                    # draw line
                    cv2.line(img, (finger_position_arr[0], finger_position_arr[1]), (bar[0], bar[1]), LINE_FROM_FINGER_COLOR, LINE_FROM_FINGER_THICKNESS)
                    # draw dots on bar also
                    cv2.circle(img, (bar[0], bar[1]), FINGER_DOT_SIZE, FINGER_DOT_COLOR, FINGER_DOT_SHAPE)

                # return the ready distance and also the bar label
                return (distance, bar[2])

    return (math.inf, "")




def main():

    """ initialize variable"""
    cap = cv2.VideoCapture(CVID)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0

    detector = htm.handDetector(min_detection_confidence=0.8)
    bar_label = ""  # a global variable to know which piano bar it is touching
    trottle_control = {i: [] for i in FINGERTIPS}  # a global variable to store conservative bars, for each finger. It use to prevent keep firing the same key when a finger stay in touching position

    pressed_bar_distance_info = dict() # initialize, info for calling piano_bar function

    while True:
        # time.sleep(0.1)
        success, img = cap.read() # initialize cv
        img = cv2.flip(img, 1) # mirror the image so that it is normal facing
        img = detector.findHands(img) # self create drawing hands class
        lmList = detector.findPosition(img, handNum=NUMBER_OF_HANDS, draw=False)

        # draw finger pt and lines
        if len(lmList) != 0:
            for finger in FINGERTIPS:
                x2, y2 = lmList[finger][1], lmList[finger][2]
                if SHOW_FINGERTIP_DOTS_AND_LINES:
                    cv2.circle(img, (x2, y2), FINGER_DOT_SIZE, (125, 125, 0), FINGER_DOT_SHAPE)

                dist, bar_label = finger_to_keys_distance(img, [x2, y2]) # return the distance and draw the line if distance is short enough

                if bar_label == "":
                    # finger is not on a bar or left the bar, clear the trottle control:
                    trottle_control = {i: [] for i in FINGERTIPS}
                    pressed_bar_distance_info = dict()

                if dist <= TOUCHING_ZONE_FROM_PIANO_BAR_CENTER:
                    # consider touching
                    print(f"finger {finger} is touched a bar {bar_label} with distance: {dist}")

                    # check if it is a conservative keys. Not playing sound if the finger is keep staying
                    if throttle_controller(trottle_control, finger, bar_label):
                        print(f"firing sound {bar_label}")
                        soundPool.submit(play_sound, bar_label)
                    pressed_bar_distance_info[bar_label] = dist

        # draw all the piano bars in one loop after knowing which bar(s) are down and also the bar y-axis postion
        for bar in PIANO_BARS:
            if bar[2] in pressed_bar_distance_info:
                # A bar is pressed. An update of bar position is needed
                draw_single_piano_bar(img, bar[2], int(pressed_bar_distance_info[bar[2]]))
            else:
                # Bar is in untouched position
                draw_single_piano_bar(img, bar[2], None)



        # fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (FPS_X_LOCATION, FPS_Y_LOCATION), FPS_FONT, FPS_FONT_SCALE,
                    FPS_FONT_COLOR, FPS_FONT_THINKNESS)

        # show cv
        cv2.imshow("Img", img)
        cv2.waitKey(CV_DELAY)

if __name__ == "__main__":
    main()