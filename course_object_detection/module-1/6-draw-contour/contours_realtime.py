import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 58, 95])
    upper_green = np.array([149, 220, 245])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw Contours
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 3)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()