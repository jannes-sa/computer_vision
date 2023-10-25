import cv2
import numpy as np

# Load Camera
cap = cv2.VideoCapture("multiple_objects.avi")

# Create mask for blue color
low_blue = np.array([80, 77, 84])
high_blue = np.array([167, 255, 255])

# Store the center positions
center_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks with color ranges
    mask = cv2.inRange(hsv_img, low_blue, high_blue)

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:
            # Draw rectangle bounding the objects detected
            (x, y, w, h) = cv2.boundingRect(cnt)

            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            center_points.append((cx, cy))
            if index == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

            index += 1



    # Display center positions
    print(center_points)
    for point in center_points:
        cv2.circle(frame, point, 5, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()