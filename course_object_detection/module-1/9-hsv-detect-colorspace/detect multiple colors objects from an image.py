import cv2
import numpy as np

# Load image
img = cv2.imread("blue_red_ball.jpg")

# Convert image to HSV color format
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create mask for red color
low_red = np.array([148, 81, 0])
high_red = np.array([179, 255, 255])
mask_red = cv2.inRange(hsv_img, low_red, high_red)

# Create mask for blue color
low_blue = np.array([80, 77, 84])
high_blue = np.array([167, 255, 255])
mask_blue = cv2.inRange(hsv_img, low_blue, high_blue)

# Link together the Masks
mask = cv2.add(mask_blue, mask_red)

# Find Contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 2000:
        # Draw precise boundaries of the objects
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)

        # Draw rectangle bounding the Object
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Show image and masks
cv2.imshow("Mask", mask)
cv2.imshow("Image", img)
cv2.imshow("Red mask", mask_red)
cv2.imshow("Blue mask", mask_blue)
cv2.waitKey(0)