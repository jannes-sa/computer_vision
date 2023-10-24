from engine.object_detection import ObjectDetection
import cv2

# 1. Load Object Detection Model
od = ObjectDetection("dnn_model/yolov8m.pt")
od.load_class_names("dnn_model/classes.txt")

# 2. Load Image
img = cv2.imread("demo/office.jpg")
bboxes, class_ids, scores = od.detect(img, imgsz=640, conf=0.25, device="cpu")

for bbox, class_id, score in zip(bboxes, class_ids, scores):
    # 1. Draw Bounding Box
    (x, y, x2, y2) = bbox
    cv2.rectangle(img, (x, y), (x2, y2), od.colors[class_id], 2)

    # 2. Get class name
    class_id = int(class_id)
    class_name = od.classes[class_id]

    # 3. Draw label
    cv2.putText(img, class_name + " " + str(score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, od.colors[class_id], 2)



cv2.imshow("image", img)
cv2.waitKey(0)