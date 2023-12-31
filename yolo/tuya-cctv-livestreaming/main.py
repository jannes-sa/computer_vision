import cv2
from ultralytics import YOLO
import numpy as np

from mapclasses import *
# map_data_cls = mapclasses.create_map_from_file('classes/classes_face_emotion.txt')
map_data_cls = mapclasses.create_map_from_file('classes/classes.txt')

import torch
print("apple m1 gpu available : ",torch.backends.mps.is_available()) # this to make sure your hardware can use M1 apple GPU to improve render performance

video_url = "rtsps://echo:6DO4FD13jnj8OrY9Qp1Ye2rcnYJjJ9zh@wework-2-us.stream.iot-11.com:443/v1/ebff52fb52e74ccfbd4ded/cljagm82v0mrt99gpkugnLliHOokIIB3?signInfo=npTNk3EnZh-0xO_unH8grFEUbjfLE3UyoF8uv6kx-X5ByZbT4OOEP7TeBUiFAnisPa2PxDq0fapS1nXe8xHS9vJE7BW6xliQWwYGJw-oM3pJ5RFLel12Q51x3lxJnIlt0XoYDq2GzrUmeYBfS63kUkZzYWQF7nfP0fHlPCbqfn0"
cap = cv2.VideoCapture(video_url)
#cap = cv2.VideoCapture(1) # open webcam
if not cap.isOpened():
    print("Error: Couldn't open the webcam / file.")
    exit()

model = YOLO("yolov8m.pt")
# model = YOLO("/Users/jannessantoso/work/ML-model-results/face-detection/weights/best.pt")

# 3) Save a Video

# Get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame_width : ", frame_width, "frame_height : ",frame_height)

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./save_video/video_saved-2.avi',fourcc, 20.0, (frame_width, frame_height))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            out.release()
            break

        # Start : Code for implement model YOLO

        # results = model(frame) # for default render using CPU performance is low
        results = model(frame, device="mps") # this using M1 Apple for render (note: you can change device into ```mps=m1 apple, cpu=for CPU, 0=for GPU```` or CPU if hardware compatible)
        # print("All Results Model ==>", results) # check result from model
        result = results[0]
        tensorBboxes = result.boxes.xyxy
        # print("TensorBboxes ==>", tensorBboxes) # bounding box which data type still in tensor

        bboxes = np.array(tensorBboxes.cpu(), dtype="int")
        # print("bboxes ==>", bboxes) # bounding box that data already normalize using dtype int

        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        # print("conf => ",result.boxes.conf)

        confidence = np.array(result.boxes.conf.cpu())
        # print("confidence np =>", confidence)

        for bbox, cls, conf in zip(bboxes, classes, confidence):
            (x, y, x2, y2) = bbox
            cls_val = map_data_cls.get(cls, "not_found")
            widthpx = x2-x
            heightpx = y2-y

            if cls_val != "not_found" and conf > 0.5:
                cv2.rectangle(frame, (x, y), (x2, y2), (0,0,255), 2) # cv2.rectangle function (frame, (x,y), (x2,y2), (Blue, Green, Red), 2)
                # cv2.putText(frame, str(cls), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
                label = cls_val+" c : "+str(conf)[:4]+" w:"+str(widthpx)+" h:"+str(heightpx)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)


        # End : Code for implement model YOLO
        out.write(frame)
        cv2.imshow("Img", frame)
        key = cv2.waitKey(1) # 1 mean continue, 0 mean stop per frame
        if key == 27 or key == ord("q") : # this shortcut key escape
           break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("KeyboardInterrupt detected, exiting...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()