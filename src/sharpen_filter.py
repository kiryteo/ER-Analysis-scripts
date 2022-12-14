import cv2
import numpy as np
import imageio

img = imageio.imread('')

low_vals = np.where(img < 10)
img[low_vals] = 0

kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
op = cv2.filter2D(img, -1, kernel)

cv2.imwrite('', op)

img_and = cv2.bitwise_and(img, op)
