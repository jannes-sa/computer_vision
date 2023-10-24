import cv2
import numpy as np

img = cv2.imread("livingroom.jpg")

# 1. Draw a line
cv2.line(img, (100, 300), (300, 100), (0, 255, 0), 5)

# 2. Draw a Rectangle
cv2.rectangle(img, (100, 100), (250, 250), (255, 0, 0), 3)

# 3. Draw a Circle
cv2.circle(img, (250, 250), 50, (0, 0, 255), cv2.FILLED)

# 4. Draw a Polygon
pts = np.array([(45, 10), (182, 13), (193, 73), (23, 100), (500, 123)], np.int32)
cv2.polylines(img, [pts], True, (100, 0, 100), 5)

# 5. Draw Text on an image
cv2.putText(img, "Hi and welcome to this lesson", (100, 500), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)