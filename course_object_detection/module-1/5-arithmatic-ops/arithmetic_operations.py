import cv2
import numpy as np

# 1) Adding two images
img1 = cv2.imread("modern_room.jpg")
img2 = cv2.imread("wooden_bench.jpg")

sum_images = cv2.add(img1, img2)
sum_weighted = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)


# 2) Threshold
img3 = cv2.imread("black_to_white.jpeg", cv2.IMREAD_GRAYSCALE)
_, binary_threshold = cv2.threshold(img3, 235, 255, cv2.THRESH_BINARY)
_, binaryinv_threshold = cv2.threshold(img3, 235, 255, cv2.THRESH_BINARY_INV)


# 3) Bitwise operators
img4 = cv2.imread("1_blackandwhite.png")
img5 = cv2.imread("2_whitebox_blackbg.png")
bitwise_and = cv2.bitwise_and(img4, img5)
bitwise_or = cv2.bitwise_or(img4, img5)
bitwise_xor = cv2.bitwise_xor(img4, img5)
bitwise_not = cv2.bitwise_not(img4)

#cv2.imshow("Img1", img1)
#cv2.imshow("Img2", img2)
#cv2.imshow("Result", sum_images)
#cv2.imshow("Weighted sum", sum_weighted)
#cv2.imshow("Img3", img3)
#cv2.imshow("Binary threshold", binary_threshold)
#cv2.imshow("Binary inverse threshold", binaryinv_threshold)
cv2.imshow("Img4", img4)
cv2.imshow("Img5", img5)
cv2.imshow("Bitwise and", bitwise_and)
cv2.imshow("Bitwise or", bitwise_or)
cv2.imshow("Bitwise xor", bitwise_xor)
cv2.imshow("Bitwise not", bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()