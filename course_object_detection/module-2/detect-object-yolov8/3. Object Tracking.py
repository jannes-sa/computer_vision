from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracking
import cv2
import numpy as np

# 1. Load Object Detection Model
od = ObjectDetection("dnn_model/yolov8m.pt")
od.load_class_names("dnn_model/classes.txt")

# 2. Load Image
cap = cv2.VideoCapture("../../../../data/highway_traffic.mp4")
# cap = cv2.VideoCapture("../data/highway_traffic.mp4")

# 3. Object Tracking
mot = MultiObjectTracking()
tracker = mot.ocsort(min_hits=5, #min_hits mean ID only given when frame detected minimum 5 frame in a row
                     max_age=10, #max_age mean if existing detection gone in 10 frame then new ID given
                     iou_threshold=0.3) # iou_threshold mean existing rectangle detection and when go to
                                        # next frame will check current rectangle is it 30% from prev rectangle
                                        # if yes then continue with same ID if not then new ID given

while True:
    # Get frame
    ret, img = cap.read()
    if not ret:
        break

    bboxes, class_ids, scores = od.detect(img, imgsz=640, conf=0.25, device="mps")

    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        # 1. Draw Bounding Box
        (x, y, x2, y2) = bbox
        # cv2.rectangle(img, (x, y), (x2, y2), od.colors[class_id], 2)

        # 2. Get class name
        class_id = int(class_id)
        class_name = od.classes[class_id]

        # 3. Draw label
        # cv2.putText(img, class_name + " " + str(score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, od.colors[class_id], 2)
    bboxes_ids = tracker.update(bboxes, scores, class_ids, img)

    for bbs_id in bboxes_ids:
        (x, y, x2, y2, id, class_id, score) = np.array(bbs_id)

        cv2.rectangle(img, (x, y), (x2, y2), od.colors[class_id], 2)
        class_name = od.classes[class_id]
        cv2.putText(img, "{} {}".format(id, class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, od.colors[class_id], 2)

    cv2.imshow("img", img)
    key = cv2.waitKey(0)
    if key == 27:
        break