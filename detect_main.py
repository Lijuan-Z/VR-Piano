import argparse
from detect_object import run


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=720)
    args = parser.parse_args()

    if run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight):
        print("Bottle detected, stopping detection.")
    else:
        print("No bottle detected.")


if __name__ == '__main__':
    main()
