import numpy as np
import cv2
# Creating a black screen image using numpy.zeros function
Img = np.zeros((480, 640, 3), dtype='uint8')

x1, y1 = (300, 400)
x2, y2 = (350, 450)

color = (255, 250, 255)
thickness = 3

# Using cv2.line() method to draw a diagonal green line with thickness of 9 px
image = cv2.line(Img, (x1, y1), (x2, y1), color, thickness) #t
image = cv2.line(Img, (x1, y1), (x1, y2), color, thickness) #l
image = cv2.line(Img, (x2, y1), (x2, y2), color, thickness) #r
image = cv2.line(Img, (x1, y2), (x2, y2), color, thickness) #b
# Display the image
cv2.imshow('Drawing_Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()