import cv2
import numpy as np

img = cv2.imread("objects.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)

# 1) Bounding rect
(x, y, w, h) = cv2.boundingRect(cnt)
#cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

# 2) Calculate area
area = cv2.contourArea(cnt)

# 3) Perimeter
perimeter = cv2.arcLength(cnt, True)

# 4) Centroid
M = cv2.moments(cnt)
cx = int(M["m10"] / M["m00"])
cy = int(M["m01"] / M["m00"])
cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)

cv2.imshow("Mask", mask)
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()