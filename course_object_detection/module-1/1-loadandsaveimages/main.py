import cv2
import numpy as np

# 1) Load Images
img = cv2.imread("japan.webp")
# 2) Show images
cv2.imshow("Image", img)
# 3) Save images
cv2.imwrite("save_images.jpg", img)


cv2.waitKey(0)
cv2.destroyAllWindows()
