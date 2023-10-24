import cv2
import numpy as np

# Load image
img = cv2.imread("red_ball.jpg")

# Convert image to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create mask for red color
low_red = np.array([148, 81, 0])
high_red = np.array([179, 255, 255])
mask = cv2.inRange(hsv_img, low_red, high_red)

# Find Contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 2000: # this useful so only area countour more than 2000 aja yang di draw
        print(area)
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)

cv2.imshow("Mask", mask)
cv2.imshow("Image", img)
cv2.waitKey(0)