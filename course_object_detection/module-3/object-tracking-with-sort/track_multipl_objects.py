import cv2
import numpy as np
from sort import *

# Load Camera
cap = cv2.VideoCapture("multiple_objects.avi")

# Create mask for blue color
low_blue = np.array([80, 77, 84])
high_blue = np.array([167, 255, 255])

# Create multiple object tracker
mot_tracker = Sort()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks with color ranges
    mask = cv2.inRange(hsv_img, low_blue, high_blue)

    # Save position objects detected
    detections = []

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:
            # Draw rectangle bounding the objects detected
            (x, y, w, h) = cv2.boundingRect(cnt)

            x2 = x + w
            y2 = y + h
            cv2.circle(frame, (x2, y2), 5, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), 2)

            # Save the boxes inside the array detections
            detections.append([x, y, x2, y2])

    # Track object using SORT
    if detections:
        objects_bbs_ids = mot_tracker.update(np.array(detections, np.int32))
        for obj in objects_bbs_ids:
            x, y, x2, y2, id = np.array(obj, np.int32)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, str(id), (x, y - 15), 0, 1, (255, 0, 0), 3)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(33)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()