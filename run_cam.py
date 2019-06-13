import argparse
from collections import deque

import cv2
import numpy as np
from keras.models import load_model
from src.config import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--color", type=str, choices=["green", "blue", "red"], default="green", help="Color which could be captured by camera and seen as pointer")
    parser.add_argument("-a", "--area", type=int, default=3000, help="Minimum area of captured object")
    parser.add_argument("-d", "--display", type=int, default=3, help="How long is prediction shown in second(s)")
    parser.add_argument("-s", "--canvas", type=bool, default=False, help="Display a Window containing your drawing without Background.")
    args = parser.parse_args()
    return args


def main(opt):
    # Define color range
    if opt.color == "red":  # We shouldn't use red as color for pointer, since it
        # could be confused with our skin's color under some circumstances
        color_lower = np.array(RED_HSV_LOWER)
        color_upper = np.array(RED_HSV_UPPER)
        color_pointer = RED_RGB
    elif opt.color == "green":
        color_lower = np.array(GREEN_HSV_LOWER)
        color_upper = np.array(GREEN_HSV_UPPER)
        color_pointer = GREEN_RGB
    else:
        color_lower = np.array(BLUE_HSV_LOWER)
        color_upper = np.array(BLUE_HSV_UPPER)
        color_pointer = BLUE_RGB

    # Initialize deque for storing detected points and canvas for drawing
    points = deque(maxlen=512)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Load the video from camera (Here I use built-in webcam)
    camera = cv2.VideoCapture(0)
    is_drawing = False
    is_shown = False

    # Load images for classes:
    predicted_class = None

    # Load model
    model = load_model("trained_models/digit_recog_model.h5")

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_drawing = not is_drawing
            if is_drawing:
                if is_shown:
                    points = deque(maxlen=512)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                is_shown = False
        if not is_drawing and not is_shown:
            if len(points):
                canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                # Blur image
                median = cv2.medianBlur(canvas_gs, 9)
                gaussian = cv2.GaussianBlur(median, (5, 5), 0)
                # Otsu's thresholding
                _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contour_gs, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contour_gs):
                    contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
                    x, y, w, h = cv2.boundingRect(contour)
                    image = canvas_gs[y:y + h, x:x + w]
                    image = cv2.resize(image, (28, 28))
                    image = np.array(image, dtype=np.float32)[None, None, :, :]
                    prediction = model.predict(np.moveaxis(image, 1, -1))
                    predicted_class = np.argmax(prediction[0])
                    # print(CLASSES[predicted_class]) #[DEBUGGING PURPOSE]
                    is_shown = True

        # Read frame from camera
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1) # Makes easy for USER to draw while looking at screen!
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        # Detect pixels fall within the pre-defined color range. Then, blur the image!
        mask = cv2.inRange(hsv, color_lower, color_upper)                                            # STACKOVEFLOW stuff!
        mask = cv2.erode(mask, kernel, iterations=2)                                                 # STACKOVEFLOW stuff!
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)                                        # STACKOVEFLOW stuff!
        mask = cv2.dilate(mask, kernel, iterations=1)                                                # STACKOVEFLOW stuff!
        contours, _= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       # TO DETECT Contours!

        # Check to see if any contours are found
        if len(contours):
            '''
            Take the biggest contour, since it is possible that there are other objects in front of camera
            whose color falls within the range of our pre-defined color.
            '''
            contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), YELLOW_RGB, 2)
            if is_drawing:
                M = cv2.moments(contour)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                points.appendleft(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1], points[i], BLUE_RGB, 60)
                    cv2.line(frame, points[i - 1], points[i], WHITE_RGB, 30)

        if is_shown:
            cv2.putText(frame, '> '+CLASSES[predicted_class]+' <', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pointer, 5, cv2.LINE_AA)
            
        cv2.imshow("Camera", frame)
        if opt.canvas:
            cv2.imshow("Canvas", canvas)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
