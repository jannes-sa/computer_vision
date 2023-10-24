import cv2
import numpy as np

# 1) Load Video from file
# cap = cv2.VideoCapture("dogs.mp4")

# 2) Load Video from webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Couldn't open the webcam / file.")
    exit()

# 3) Save a Video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('transport_flipped.avi',fourcc, 30.0, (1280,720))

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    height, width, channels = frame.shape

    # save frame
    flipped_frame = cv2.flip(frame, 0)
    out.write(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow("Flipped frame", flipped_frame)

    key = cv2.waitKey(33)
    if key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()