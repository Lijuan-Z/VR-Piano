import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm
import concurrent.futures
from playsound import playsound

soundPool = concurrent.futures.ThreadPoolExecutor(max_workers=20)

# screen control
FRAME_PER_SECOND = None  # 0.1 means 10 frame per second, 0.5 means 2 frame per second. Put "None" if you want the fastest output
SHOW_FINGERTIP_DOTS_AND_LINES = True
SHOW_TOP_MESSAGE = True
SHOW_FPS = True

CVID = 0
CV_DELAY = 1
wCam, hCam = 1024, 768

# FPS & MESSAGES
FPS_X_LOCATION, FPS_Y_LOCATION = 30, 40
FPS_FONT = cv2.FONT_HERSHEY_COMPLEX
FPS_FONT_SCALE, FPS_FONT_THINKNESS = 1, 2
FPS_FONT_COLOR = (255, 0, 0)
TOP_RIGHT_MSG_COLOR = (75, 0, 130)

# Finger drawing
FINGER_DOT_SIZE = 7
FINGER_DOT_COLOR = (75, 75, 75)
FINGER_DOT_SHAPE = cv2.FILLED
LINE_FROM_FINGER_COLOR = (0, 255, 255)
LINE_FROM_FINGER_THICKNESS = 2

# Size of piano bars in square. e.g. 20 = 40 width * 40 height
BAR_SIZE_FROM_CENTER = 20       # KEY FACTOR TO CONTROL THE SIZE OF PIANO BARS

# Touching point (hand)
TOUCHING_POINT_COLOR = (255, 255, 255)
TOUCHING_POINT_SHAPE = cv2.FILLED
TOUCHING_POINT_SIZE = 7

# Touching point (piano bar)
TOUCHING_ZONE_FROM_PIANO_BAR_CENTER = BAR_SIZE_FROM_CENTER  # KEY FACTOR TO DEFINE THE STARTING POINT OF TOUCH IN Y-AXIS
READY_ZONE = 80
BAR_DOWN_RANGE = (-20, TOUCHING_ZONE_FROM_PIANO_BAR_CENTER)
TOUCHING_ZONE_COLOR = (0, 0, 255)
TOUCHING_ZONE_WIDTH_DEDUCTION = 10

# Piano bar properites
PIANO_BAR_COLOR = (0, 255, 0)
PIANO_BARS_CENTER_POS = (325, 425)
PIANO_BAR_LD_CENTER = (PIANO_BARS_CENTER_POS[0] - 12 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'LD')
PIANO_BAR_LE_CENTER = (PIANO_BARS_CENTER_POS[0] - 10 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'LE')
PIANO_BAR_LF_CENTER = (PIANO_BARS_CENTER_POS[0] - 8 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'LF')
PIANO_BAR_LG_CENTER = (PIANO_BARS_CENTER_POS[0] - 6 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'LG')
PIANO_BAR_A_CENTER = (PIANO_BARS_CENTER_POS[0] - 4 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'A')
PIANO_BAR_B_CENTER = (PIANO_BARS_CENTER_POS[0] - 2 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'B')
PIANO_BAR_C_CENTER = (PIANO_BARS_CENTER_POS[0] + 0, PIANO_BARS_CENTER_POS[1], 'C')
PIANO_BAR_D_CENTER = (PIANO_BARS_CENTER_POS[0] + 2 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'D')
PIANO_BAR_E_CENTER = (PIANO_BARS_CENTER_POS[0] + 4 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'E')
PIANO_BAR_F_CENTER = (PIANO_BARS_CENTER_POS[0] + 6 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'F')
PIANO_BAR_G_CENTER = (PIANO_BARS_CENTER_POS[0] + 8 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'G')
PIANO_BAR_RA_CENTER = (PIANO_BARS_CENTER_POS[0] + 10 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'RA')
PIANO_BAR_RB_CENTER = (PIANO_BARS_CENTER_POS[0] + 12 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'RB')
PIANO_BAR_RC_CENTER = (PIANO_BARS_CENTER_POS[0] + 14 * BAR_SIZE_FROM_CENTER, PIANO_BARS_CENTER_POS[1], 'RC')

# register piano bars
PIANO_BARS = [PIANO_BAR_LD_CENTER, PIANO_BAR_LE_CENTER, PIANO_BAR_LF_CENTER, PIANO_BAR_LG_CENTER, PIANO_BAR_A_CENTER,
              PIANO_BAR_B_CENTER, PIANO_BAR_C_CENTER, PIANO_BAR_D_CENTER, PIANO_BAR_E_CENTER, PIANO_BAR_F_CENTER,
              PIANO_BAR_G_CENTER, PIANO_BAR_RA_CENTER, PIANO_BAR_RB_CENTER, PIANO_BAR_RC_CENTER, ]

# under bar note
NOTE_FONT = cv2.FONT_HERSHEY_COMPLEX
NOTE_SCALE, NOTE_THINKNESS = 0.6, 2

# FINGERTIPS = [8] # currently one finger
# FINGERTIPS = [4, 8, 12, 16, 20]
FINGERTIPS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
NUMBER_OF_HANDS = 2 # one or two hands

THROTTLE_THRESHOLD = 100     # if a finger stay at the same bar. This control how many conservative bars to wait before we allow another sound playing

def play_sound(key):
    match key:
        case 'LD':
            playsound("./sound/key05.mp3")
        case 'LE':
            playsound("./sound/key06.mp3")
        case 'LF':
            playsound("./sound/key07.mp3")
        case 'LG':
            playsound("./sound/key08.mp3")
        case 'A':
            playsound("./sound/key09.mp3")
        case 'B':
            playsound("./sound/key10.mp3")
        case 'C':
            playsound("./sound/key11.mp3")
        case 'D':
            playsound("./sound/key12.mp3")
        case 'E':
            playsound("./sound/key13.mp3")
        case 'F':
            playsound("./sound/key14.mp3")
        case 'G':
            playsound("./sound/key15.mp3")
        case 'RA':
            playsound("./sound/key16.mp3")
        case 'RB':
            playsound("./sound/key17.mp3")
        case 'RC':
            playsound("./sound/key18.mp3")
def get_positions_by_bar_name(bar):
    match bar:
        case 'LD':
            return (PIANO_BAR_LD_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_LD_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_LD_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_LD_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'LE':
            return (PIANO_BAR_LE_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_LE_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_LE_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_LE_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'LF':
            return (PIANO_BAR_LF_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_LF_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_LF_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_LF_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'LG':
            return (PIANO_BAR_LG_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_LG_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_LG_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_LG_CENTER[1] + BAR_SIZE_FROM_CENTER)
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
        case 'F':
            return (PIANO_BAR_F_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_F_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_F_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_F_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'G':
            return (PIANO_BAR_G_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_G_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_G_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_G_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'RA':
            return (PIANO_BAR_RA_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_RA_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_RA_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_RA_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'RB':
            return (PIANO_BAR_RB_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_RB_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_RB_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_RB_CENTER[1] + BAR_SIZE_FROM_CENTER)
        case 'RC':
            return (PIANO_BAR_RC_CENTER[0] - BAR_SIZE_FROM_CENTER, PIANO_BAR_RC_CENTER[1] - BAR_SIZE_FROM_CENTER, PIANO_BAR_RC_CENTER[0] + BAR_SIZE_FROM_CENTER, PIANO_BAR_RC_CENTER[1] + BAR_SIZE_FROM_CENTER)


def draw_single_piano_bar(img, bar, pos):
    # Drawing single piano bar with real-time y-axis of pressed bar
    # print(bar, bar_label)
    print(f"pos {bar}: ", pos)
    x1, y1, x2, y2 = get_positions_by_bar_name(bar)
    if pos is not None:
        # bar down
        pos = np.interp(pos, [BAR_DOWN_RANGE[0], BAR_DOWN_RANGE[1]], [y2, y1])


        # cv2.line(img, (x1, int(pos)), (x2, int(pos)), PIANO_BAR_COLOR, 3)  # top, both p1, p2 y move
        cv2.line(img, (x1, y1), (x1, int(pos)), PIANO_BAR_COLOR, 3)  # left,  y2 move
        cv2.line(img, (x2, y1), (x2, int(pos)), PIANO_BAR_COLOR, 3)  # right, y2 move
        cv2.line(img, (x1, y2), (x2, y2), PIANO_BAR_COLOR, 3)  # bottom, never move
        cv2.rectangle(img, (x1, int(pos)), (x2, y2), PIANO_BAR_COLOR, cv2.FILLED) # the filled cube that show when finger is on

    else:
        # No touching, draw normal bar
        cv2.line(img, (x1, y1), (x2, y1), PIANO_BAR_COLOR, 3)  # top, both p1, p2 y move
        cv2.line(img, (x1, y1), (x1, y2), PIANO_BAR_COLOR, 3)  # left,  y2 move
        cv2.line(img, (x2, y1), (x2, y2), PIANO_BAR_COLOR, 3)  # right, y2 move
        cv2.line(img, (x1, y2), (x2, y2), PIANO_BAR_COLOR, 3)  # bottom, never move

    # Put the letter under each key
    cv2.putText(img, bar, (x1 + 5, y2 + 20), NOTE_FONT, NOTE_SCALE, PIANO_BAR_COLOR, NOTE_THINKNESS)
    return pos

def throttle_controller(trottle_control, finger, bar):
    # True: wait enough time, allow to play the sound
    # False: not enough waiting, don't allow any sound play
    # print(trottle_control)

    if len(trottle_control[finger]) == 0:
        # the first bar, always allow to play conservatively
        trottle_control[finger].append(bar)
        # return True
    elif len(trottle_control[finger]) < THROTTLE_THRESHOLD:
        if trottle_control[finger][-1] == bar:
            # only add bar if the current bar is the same as last bar
            trottle_control[finger].append(bar)
            # return False
        else:
            # otherwise it is another bar and we should allow and clear the list
            trottle_control[finger] = []
            # return True
    else:
        # The bar is already pass threshold, allow to play
        trottle_control[finger] = []
        # return True

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
        # Staring by consider only finger position within the range of current bar, and restricted margin
        if finger_position_arr[0] > LEFT_TOUCHING_MARGIN and finger_position_arr[0] < RIGHT_TOUCHING_MARGIN:
            # print(f"{bar[2]} is testing range {LEFT_TOUCHING_MARGIN} > {finger_position_arr[0]} < {RIGHT_TOUCHING_MARGIN}")
            if SHOW_FINGERTIP_DOTS_AND_LINES:
                cv2.rectangle(img, (LEFT_TOUCHING_MARGIN, PIANO_BARS_CENTER_POS[1] - TOUCHING_ZONE_FROM_PIANO_BAR_CENTER), (RIGHT_TOUCHING_MARGIN, PIANO_BARS_CENTER_POS[1]), TOUCHING_ZONE_COLOR, cv2.FILLED)

        # if within the bar width, calculate distance, draw line if within ready zone

            distance = math.hypot(finger_position_arr[0] - bar[0], finger_position_arr[1] - bar[1])
            if bar[1] - finger_position_arr[1] < 0:
                distance = -distance

            # print(f"in key {bar[2]}, {distance}")
            if distance <= READY_ZONE:
                print(f"READY_ZONE {bar[2]}, {distance}")

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
    # bar_label = ""  # a global variable to know which piano bar it is touching
    trottle_control = {i: [] for i in FINGERTIPS}  # a global variable to store conservative bars, for each finger. It use to prevent keep firing the same key when a finger stay in touching position
    isBarEnabled = {j[2]: True for j in PIANO_BARS} # a variable to store the whether a bar is enabled or disabled, prevent multiple fingers pressing the same bar and play sound

    pressed_bar_distance_info = dict() # initialize, info for calling piano_bar function

    while True:
        if FRAME_PER_SECOND:
            time.sleep(FRAME_PER_SECOND)
        success, img = cap.read() # initialize cv
        img = cv2.flip(img, 1) # mirror the image so that it is normal facing
        img = detector.findHands(img) # self create drawing hands class
        lmList = detector.findPosition(img, handNum=NUMBER_OF_HANDS, draw=False)



        # draw finger pt and lines
        if len(lmList) != 0:
            CURRENTFINGERTIPS = FINGERTIPS
            if len(lmList) <= 21:
                # only one hand detected
                CURRENTFINGERTIPS = FINGERTIPS[:5]

            for finger in CURRENTFINGERTIPS:
                x2, y2 = lmList[finger][1], lmList[finger][2]
                if SHOW_FINGERTIP_DOTS_AND_LINES:
                    cv2.circle(img, (x2, y2), FINGER_DOT_SIZE, (125, 125, 0), FINGER_DOT_SHAPE)

                dist, bar_label = finger_to_keys_distance(img, [x2, y2]) # return the distance and draw the line if distance is short enough

                print(finger, isBarEnabled, trottle_control)

                # Current finger Entering a bar
                if dist <= TOUCHING_ZONE_FROM_PIANO_BAR_CENTER:
                    # consider touching
                    # print(f"finger {finger} is touched a bar {bar_label} with distance: {dist}")
                    pressed_bar_distance_info[bar_label] = dist


                    # Tracking conservative bars. Not playing sound if the finger is keep staying
                    throttle_controller(trottle_control, finger, bar_label)

                    # only play a sound if the bar is enabled
                    if isBarEnabled[bar_label]:
                        print(f"firing sound {bar_label}")
                        soundPool.submit(play_sound, bar_label)
                    isBarEnabled[bar_label] = False

                # Current finger Leaving a bar
                else:
                    # distance is more than TOUCHING_ZONE_FROM_PIANO_BAR_CENTER and current finger was previously touched
                    trottle_control[finger] = []
                    print(f'{finger} is leaving {bar_label}" is enabled')
                    isBarEnabled[bar_label] = True
                    pressed_bar_distance_info[bar_label] = None

                # cv2.putText(img, f'finger {finger} distance: {dist}', (40, FPS_Y_LOCATION + 60), FPS_FONT, 0.4, TOP_RIGHT_MSG_COLOR, 1) #tmp
                # cv2.putText(img, f'finger {finger} pressed_bar_distance_info: {pressed_bar_distance_info}', (40, FPS_Y_LOCATION + 80), FPS_FONT, 0.4, TOP_RIGHT_MSG_COLOR, 1) #tmp



        # draw all the piano bars in one loop after knowing which bar(s) are down and also the bar y-axis postion
        bar_static = []
        for bar in PIANO_BARS:
            bar_adjusted_pos = None
            if bar[2] in pressed_bar_distance_info:
                # A bar is pressed. An update of bar position is needed
                bar_adjusted_pos = draw_single_piano_bar(img, bar[2], pressed_bar_distance_info[bar[2]])
            else:
                # Bar is in untouched position
                bar_adjusted_pos = draw_single_piano_bar(img, bar[2], None)

            bar_static.append((bar[2], bar_adjusted_pos))

        if SHOW_TOP_MESSAGE:
            show_string = "Bar Pressed: "
            y_increment = 20
            for k, v in isBarEnabled.items():
                if v == True:
                    show_string += f"{k}:T "
                else:
                    show_string += f"{k}:F "
                cv2.putText(img, show_string, (40, FPS_Y_LOCATION + y_increment), FPS_FONT, 0.4, TOP_RIGHT_MSG_COLOR, 1)

            show_string = "Piano Bar Y-Axis: "
            y_increment = 40
            for idx, item in enumerate(bar_static):
                if item[1] is not None:
                    show_string += f"{item[0]}: {item[1]:.1f} "
                else:
                    show_string += f"{item[0]}: {item[1]} "
                if idx % 6 == 0:
                    if idx != 0:
                        cv2.putText(img, show_string, (40, FPS_Y_LOCATION + y_increment), FPS_FONT, 0.4, TOP_RIGHT_MSG_COLOR, 1)
                    show_string = ""
                    y_increment += 20



        # fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        if SHOW_FPS:
            cv2.putText(img, f'FPS: {int(fps)}', (FPS_X_LOCATION, FPS_Y_LOCATION), FPS_FONT, FPS_FONT_SCALE,
                    FPS_FONT_COLOR, FPS_FONT_THINKNESS)

        # show cv
        cv2.imshow("Img", img)
        cv2.waitKey(CV_DELAY)

if __name__ == "__main__":
    main()