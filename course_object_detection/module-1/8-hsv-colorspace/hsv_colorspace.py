import cv2
import numpy as np

def nothing(x):
    pass

# Create Trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Low - H", "Trackbars", 102, 179, nothing)
cv2.createTrackbar("Low - s", "Trackbars", 102, 255, nothing)
cv2.createTrackbar("Low - v", "Trackbars", 102, 255, nothing)
cv2.createTrackbar("High - H", "Trackbars", 102, 179, nothing)
cv2.createTrackbar("High - s", "Trackbars", 102, 255, nothing)
cv2.createTrackbar("High - v", "Trackbars", 102, 255, nothing)

# Load image image
img = cv2.imread("colors.jpg")

# Convert image to HSV color format
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    # Opencv HSV ranges H = 0-179, S = 0-255, V = 0-255
    l_h = cv2.getTrackbarPos("Low - H", "Trackbars")
    l_s = cv2.getTrackbarPos("Low - s", "Trackbars")
    l_v = cv2.getTrackbarPos("Low - v", "Trackbars")
    h_h = cv2.getTrackbarPos("High - H", "Trackbars")
    h_s = cv2.getTrackbarPos("High - s", "Trackbars")
    h_v = cv2.getTrackbarPos("High - v", "Trackbars")

    # HSV lower and upper ranges
    low_blue = np.array([l_h, l_s, l_v])
    high_blue = np.array([h_h, h_s, h_v])
    mask = cv2.inRange(hsv_img, low_blue, high_blue)

    # Show only the elements not covered by the mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show image and mask
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()