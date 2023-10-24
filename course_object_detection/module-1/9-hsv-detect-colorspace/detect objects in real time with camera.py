import cv2
import numpy as np

# Load Camera
cap = cv2.VideoCapture(1)

# Create mask for red color
low_red = np.array([148, 81, 0])
high_red = np.array([179, 255, 255])

# Create mask for blue color
low_blue = np.array([80, 77, 84])
high_blue = np.array([167, 255, 255])

while True:
    _, frame = cap.read()
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks with color ranges
    mask_red = cv2.inRange(hsv_img, low_red, high_red)
    mask_blue = cv2.inRange(hsv_img, low_blue, high_blue)

    # Link together the Masks
    mask = cv2.add(mask_blue, mask_red)

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 2000:
            # Draw precise boundaries of the objects
            # cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)

            # Draw rectangle bounding the objects detected
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()