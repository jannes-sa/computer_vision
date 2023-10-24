from engine.object_detection import ObjectDetection
from engine.object_tracking import MultiObjectTracking
import cv2
import numpy as np

# 1. Load Object Detection
od = ObjectDetection("dnn_model/yolov8m.pt")
od.load_class_names("dnn_model/classes.txt")

# 2. Videocapture
cap = cv2.VideoCapture("demo/highway.mp4")

# 3. Load tracker
mot = MultiObjectTracking()
tracker = mot.ocsort()

# 4. Store Trajectories
trajectory_by_id = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. We detect the objects
    bboxes, class_ids, scores = od.detect(frame, imgsz=640, conf=0.25)

    # 2. We track the objects detected
    bboxes_ids = tracker.update(bboxes, scores, class_ids, frame)
    for bbox_id in bboxes_ids:
        (x, y, x2, y2, object_id, class_id, score) = np.array(bbox_id)

        # Center point
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        # Draw center point
        cv2.circle(frame, (cx, cy), 5, (15, 25, 215), -1)

        # draw bounding box
        cv2.rectangle(frame, (x, y), (x2, y2), od.colors[class_id], 2)

        # Object id
        cv2.putText(frame, "{}".format(object_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.4, od.colors[class_id], 2)

        # Add first point of object ID once detected
        if object_id not in trajectory_by_id:
            trajectory_by_id[object_id] = [(cx, cy)]
        else:
            trajectory_by_id[object_id].append((cx, cy))

            # Draw Trajectory
            trajectory = trajectory_by_id[object_id]
            cv2.polylines(frame, [np.array(trajectory[-20:])], False, (15, 225, 215), 2)
            print("ID : {}, Trajectory: {}".format(object_id, trajectory))


    print("Trajectories by ID: ", trajectory_by_id)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()