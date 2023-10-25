import cv2
import numpy as np
import math

# Load Camera
cap = cv2.VideoCapture("multiple_objects.avi")

# Create mask for blue color
low_blue = np.array([80, 77, 84])
high_blue = np.array([167, 255, 255])

# Store the center positions
center_points = {}
id_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks with color ranges
    mask = cv2.inRange(hsv_img, low_blue, high_blue)

    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:
            # Draw rectangle bounding the objects detected
            (x, y, w, h) = cv2.boundingRect(cnt)

            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            #print(cx, cy)

            same_object_detected = False
            for id, pt in center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 20:
                    center_points[id] = (cx, cy)
                    same_object_detected = True
                    cv2.putText(frame, str(id), (x, y - 15), 0, 1, (255, 0, 0), 2)
                    continue

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                center_points[id_count] = (cx, cy)
                id_count += 1


            #center_points.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)



    # Display center positions


    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()