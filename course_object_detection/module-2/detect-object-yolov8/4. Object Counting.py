from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracking
import cv2
import numpy as np

# 1. Load Object Detection
od = ObjectDetection("dnn_model/yolov8m.pt")
od.load_class_names("dnn_model/classes.txt")

# 2. Videocapture
# cap = cv2.VideoCapture("demo/highway.mp4")
cap = cv2.VideoCapture("../../../../data/highway_traffic.mp4")

# 3. Load tracker
mot = MultiObjectTracking()
tracker = mot.ocsort(min_hits=5, #min_hits mean ID only given when frame detected minimum 5 frame in a row
                      max_age=10, #max_age mean if existing detection gone in 10 frame then new ID given
                      iou_threshold=0.3)

# Object Counting Area
crossing_area_1 = np.array([(343, 22), (584,13), (632, 42), (624, 185), (313, 121)])
crossing_area_2 = np.array([(12, 183), (6, 25), (290, 23), (280, 188)])
vehicles_ids1 = set()
vehicles_ids2 = set()

x_start, y_start, x_end, y_end = 1, 164, 637, 356

while True:
    ret, frame = cap.read()
    if not ret:
        break


    cropped_frame = frame[y_start:y_end, x_start:x_end]
    cv2.putText(frame, "Area Start Detect", (x_start, y_start - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),1)
    cv2.rectangle(frame, (x_start, y_start),(x_end, y_end), (0,255,0), 1)

    # 1. We detect the objects
    bboxes, class_ids, scores = od.detect(cropped_frame, imgsz=640, conf=0.25, device="mps")

    # 2. We track the objects detected
    bboxes_ids = tracker.update(bboxes, scores, class_ids, cropped_frame)
    for bbox_id in bboxes_ids:
        (x, y, x2, y2, object_id, class_id, score) = np.array(bbox_id)
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        # draw bounding box
        # cv2.rectangle(frame, (x, y), (x2, y2), od.colors[class_id], 2)
        cv2.circle(cropped_frame, (cx, cy), 5, od.colors[class_id], -1)

        # Object id
        cv2.putText(cropped_frame, "{}".format(object_id), (cx, cy - 10), cv2.FONT_HERSHEY_PLAIN, 1.4, od.colors[class_id], 2)

        is_inside1 = cv2.pointPolygonTest(crossing_area_1, (cx, cy), False)
        is_inside2 = cv2.pointPolygonTest(crossing_area_2, (cx, cy), False)
        # If the object is inside the crossing area
        if is_inside1 > 0:
            vehicles_ids1.add(object_id)
        elif is_inside2 > 0:
            vehicles_ids2.add(object_id)



    cv2.putText(frame, "VEHICLES AREA 1: {}".format(len(vehicles_ids1)), (400, 50), cv2.FONT_HERSHEY_PLAIN,
                1, (15, 225, 215), 1)
    cv2.putText(frame, "VEHICLES AREA 2: {}".format(len(vehicles_ids2)), (10, 50), cv2.FONT_HERSHEY_PLAIN,
                1, (15, 225, 215), 1)

    # Draw area
    cv2.polylines(cropped_frame, [crossing_area_1], True, (15, 225, 215), 2)
    cv2.polylines(cropped_frame, [crossing_area_2], True, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Cropped Frame", cropped_frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()