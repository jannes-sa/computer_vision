import cv2
import numpy as np

# 1) What are images
img = cv2.imread("plant.jpg", cv2.IMREAD_GRAYSCALE)

# - Creat our own grayscale image
our_img = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 255, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 101, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 30, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 220, 0],
    ]
our_img = np.array(our_img, np.uint8)

# Color Image
img_color = cv2.imread("plant.jpg")

# Creat our color image
our_color_img = [
    [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
    [[0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0]],
    [[0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255]],
    [[0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255]],
    [[0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255]],
    [[0, 200, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255]],
    [[0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255]],
    ]
our_color_img = np.array(our_color_img, np.uint8)



# 2) Access image info, access pixels and edit them
rows, columns, channels = img_color.shape
pixel_value = img_color[320, 320]

img_color[50, 320] = [0, 0, 255]


# 3) Select a region of interest ROI
roi = img_color[200: 440, 200: 440]
roi_rows, roi_col, _ = roi.shape
print(roi_rows, roi_col)

# Edit a region of intereste
img_color[0: 240, 0: 240] = roi

cv2.imshow("Our img", our_img)
cv2.imshow("Our img color", our_color_img)
cv2.imshow("Color img", img_color)
cv2.imshow("Roi", roi)
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()