import cv2
from ultralytics import YOLO
import numpy as np

from mapclasses import *
map_data_cls = mapclasses.create_map_from_file('classes/classes.txt')

import torch
print("apple m1 gpu available : ",torch.backends.mps.is_available()) # this to make sure your hardware can use M1 apple GPU to improve render performance

cap = cv2.VideoCapture("../../../data/highway_traffic.mp4")
if not cap.isOpened():
    print("Error: Couldn't open the webcam / file.")
    exit()

model = YOLO("yolov8m.pt")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, conf=0.25, iou=0.5, persist=True, device="mps") # this using M1 Apple for render (note: you can change device into ```mps=m1 apple, cpu=for CPU, 0=for GPU```` or CPU if hardware compatible)
        result = results[0]
        annotated_frame = result.plot()
        tensorBboxes = result.boxes.xyxy

        bboxes = np.array(tensorBboxes.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        track_ids = np.array(result.boxes.id.int().cpu().tolist(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")

        for box, track_id, score, class_id in zip(bboxes, track_ids, scores, class_ids):
            x, y, x2, y2 = box
            cx = int((x + x2) / 2)
            cy = int((y + y2) / 2)
            print("Track ID", track_id, "Box Data", box)

            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
            cv2.putText(frame, "{} {}".format(track_id, class_id), (cx,cy-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1)

        # End : Code for implement model YOLO
        cv2.imshow("Frame", frame)
        # cv2.imshow("Anotated Frame", annotated_frame)
        key = cv2.waitKey(0) # 1 mean continue, 0 mean stop per frame
        if key == 27 or key == ord("q") : # this shortcut key escape
           break

    cap.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("KeyboardInterrupt detected, exiting...")
    cap.release()
    cv2.destroyAllWindows()