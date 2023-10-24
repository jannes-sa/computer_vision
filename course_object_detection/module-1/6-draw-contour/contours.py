import cv2
import numpy as np

img = cv2.imread("human_hand.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

# Find the contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw contours
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cv2.imshow("Img", img)
cv2.imshow("Gray", gray)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()