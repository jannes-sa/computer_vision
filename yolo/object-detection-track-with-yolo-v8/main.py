import cv2
from ultralytics import YOLO
import numpy as np
import time

from mapclasses import *
map_data_cls = mapclasses.create_map_from_file('classes/classes.txt')

import torch
print("apple m1 gpu available : ",torch.backends.mps.is_available()) # this to make sure your hardware can use M1 apple GPU to improve render performance

cap = cv2.VideoCapture("../../../data/highway_traffic.mp4")
if not cap.isOpened():
    print("Error: Couldn't open the webcam / file.")
    exit()

model = YOLO("cnn_model/yolov8m.pt")
vehicles_ids = set()

# coordinate for cropped frame 2
x_start, y_start, x_end, y_end = 332, 171, 638, 351
cy1 = 15
cy2 = 80
offset = 6
dict_id_with_speed = {}
distance1 = 25 # meters

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[y_start:y_end, x_start:x_end]

        results = model.track(cropped_frame, conf=0.25, iou=0.5, persist=True, device="mps") # this using M1 Apple for render (note: you can change device into ```mps=m1 apple, cpu=for CPU, 0=for GPU```` or CPU if hardware compatible)
        result = results[0]
        annotated_frame = result.plot()
        tensorBboxes = result.boxes.xyxy

        bboxes = np.array(tensorBboxes.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        try:
            track_ids = np.array(result.boxes.id.int().cpu().tolist(), dtype="int")
        except AttributeError:
            track_ids = np.array([], dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")

        # : START Loop Model
        for box, track_id, score, class_id in zip(bboxes, track_ids, scores, class_ids):
            x, y, x2, y2 = box
            cx = int((x + x2) / 2)
            cy = int((y + y2) / 2)
            print("Track ID", track_id, "Box Data", box)
            if track_id:
                vehicles_ids.add(track_id)

            cv2.circle(cropped_frame, (cx, cy), 5, (0,0,255), -1)
            cv2.putText(cropped_frame, "{} {}".format(track_id, map_data_cls[class_id]), (cx,cy-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1)

            if cy1 < (cy + offset-4) and cy1 > (cy - offset-4):
                now_cy1 = time.time()
                dict_id_with_speed[track_id] = now_cy1
                cv2.putText(cropped_frame,"detect",(cx, cy-3), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),1)
                print("touch cy1 => ", track_id, " with time => ", now_cy1)
            if track_id in dict_id_with_speed:

                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    now_cy2 = time.time()
                    elapsed1_time = now_cy2 - dict_id_with_speed[track_id]
                    distance_km = distance1/1000 # calculate in km
                    time_hours = elapsed1_time/3600 # calculate in hour
                    speed_kmh = distance_km/time_hours

                    #a_speed_ms1 = distance1 / elapsed1_time
                    #a_speed_kh1 = a_speed_ms1 * 3.6

                    cv2.putText(cropped_frame, "{} km/h".format(int(speed_kmh)), (cx, cy - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                    print("touch cy2 => ", track_id, "before time => ",dict_id_with_speed[track_id], " with time => ", now_cy2, " and elapse time => ", elapsed1_time)

        # : END Loop Model

        cv2.putText(cropped_frame, "VEHICLES AREA 1: {}".format(len(vehicles_ids)), (400, 50), cv2.FONT_HERSHEY_PLAIN,
                   1, (15, 225, 215), 1)

        cv2.line(cropped_frame, (1, cy1), (305, cy1), (255, 255, 255), 1)
        cv2.line(cropped_frame, (1, cy2), (305, cy2), (255, 255, 255), 1)

        # End : Code for implement model YOLO
        cv2.imshow("Frame Original", frame)
        cv2.imshow("Frame Cropped 2", cropped_frame)

        # cv2.imshow("Anotated Frame", annotated_frame)
        key = cv2.waitKey(1) # 1 mean continue, 0 mean stop per frame
        if key == 27 or key == ord("q") : # this shortcut key escape
           break

    cap.release()
    cv2.destroyAllWindows()


except KeyboardInterrupt:
    print("KeyboardInterrupt detected, exiting...")
    cap.release()
    cv2.destroyAllWindows()