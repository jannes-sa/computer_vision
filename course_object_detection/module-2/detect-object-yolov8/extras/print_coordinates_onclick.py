import cv2

cap = cv2.VideoCapture("../demo/highway.mp4")

def print_coordinates(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates: x: {}, y:{}".format(x, y))

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", print_coordinates)
while True:
    ret, frame = cap.read()
    if not ret:
        # Restart video
        # set frame to 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue




    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()