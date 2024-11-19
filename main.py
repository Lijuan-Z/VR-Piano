import pickle
import time

import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
def print_hand():
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)
    pTime = 0 #previous time
    cTime = 0 #current time
    while True:
        ret, img = cap.read()
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)
            # print(result.multi_hand_landmarks)
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
                    for i,lm in enumerate(handLms.landmark):
                        xPos = lm.x*img.shape[1]
                        yPos = lm.y*img.shape[0]
                        print(i,xPos,yPos)
            cTime = time.time()
            fps = 1 / (cTime - pTime)  #depends on computer and with hands less than without hands, maybe 15~20, means how many image one second?
            pTime = cTime
            cv2.putText(img,f"FPS: {int(fps)}",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
            cv2.imshow('img',img)
        if cv2.waitKey(1) == ord('q'):
            break

def video_processing():
    import cv2
    import mediapipe as mp

    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Open a video capture or use an image.
    cap = cv2.VideoCapture("twinkle_twinkle_little_star.mp4")  # Use 0 for webcam, or a file path for video

    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)

    pTime = 0 #previous time
    cTime = 0 #current time
    landmark_data = {i: {'x': [], 'y': [], 'z': []} for i in range(21)}
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands.
        results = hands.process(image_rgb)

        # Draw hand landmarks.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
                for i,lm in enumerate(hand_landmarks.landmark):
                    xPos = int(lm.x*image.shape[1])
                    yPos = int(lm.y*image.shape[0])
                    # # print(i,xPos,yPos)
                    # print(i,lm.x,lm.y,lm.z)
                    landmark_data[i]['x'].append(lm.x)
                    landmark_data[i]['y'].append(lm.y)
                    landmark_data[i]['z'].append(lm.z)
                    cv2.putText(image, str(i), (xPos-25,yPos+5), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),2,)
        with open('landmark_data_twinkle_youtube.pkl', 'wb') as f:
            pickle.dump(landmark_data, f)
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # depends on computer and with hands less than without hands, maybe 15~20, means how many image one second?
        pTime = cTime
        cv2.putText(image, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('img', image)

        # Display the output.
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot the (x, y, z) coordinates for each landmark in subplots.
    fig, axes = plt.subplots(7, 3, figsize=(15, 20))
    axes = axes.flatten()

    for i in range(21):
        ax = axes[i]
        ax.plot(landmark_data[i]['x'], label='x', color='red')
        ax.plot(landmark_data[i]['y'], label='y', color='green')
        ax.plot(landmark_data[i]['z'], label='z', color='blue')
        ax.set_title(f'Landmark {i}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def data_processing():
    with open('landmark_data.pkl', 'rb') as f:
        landmark_data = pickle.load(f)  # This should be a dictionary of lists

    selected_landmarks_z = {index: [] for index in [4, 8, 12, 16, 20]}

    # Iterate through frames stored in each landmark index
    for i in selected_landmarks_z.keys():
        selected_landmarks_z[i] = landmark_data[i]['z']

    # Plot the `z` data for each selected landmark
    plt.figure(figsize=(10, 6))
    for idx, z_values in selected_landmarks_z.items():
        plt.plot(z_values, label=f'Landmark {idx}')

    plt.title('Z Coordinates of Selected Landmarks Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Z Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def gpt_example():
    # Setup MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Define the piano key positions (simplified)
    piano_keys = [
        (50, 100),  # Position for C
        (100, 100),  # Position for D
        (150, 100),  # Position for E
        # Add more keys here...
    ]

    # Initialize camera
    cap = cv2.VideoCapture("twinkle_twinkle_little_star.mp4")
    # cap = cv2.VideoCapture("fur elise.mov")# Or path to your video file

    # Time tracking for FPS
    pTime = 0  # previous time
    cTime = 0  # current time

    while True:
        ret, image = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for i, lm in enumerate(hand_landmarks.landmark):
                    # Get x, y positions of landmarks
                    xPos = int(lm.x * image.shape[1])
                    yPos = int(lm.y * image.shape[0])

                    # Assume index finger landmark 8 corresponds to pressing a key
                    if i == 8:  # Landmark 8 corresponds to index finger tip
                        for idx, (key_x, key_y) in enumerate(piano_keys):
                            # A simple threshold check for detecting key press (yPos threshold to detect pressing)
                            if abs(xPos - key_x) < 30 and yPos > key_y:  # 30 is the threshold for x-position accuracy
                                print(f"Key {idx} pressed (x: {xPos}, y: {yPos})")
                                cv2.putText(image, f'Key {idx} pressed', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f"FPS: {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Show the video feed with the drawn landmarks
        cv2.imshow("Hand Tracking - Piano", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    video_processing()
    # data_processing()
    # gpt_example()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
