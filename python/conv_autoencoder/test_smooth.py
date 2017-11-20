import cv2
import numpy as np

path = '/home/srq/Datasets/tables/uw3/correctgt-zones/D05DBIN.png'

image = cv2.imread(path,0)

height, width = np.shape(image)

scale_factor = 500 / max(height,width)

image_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

K = cv2.getGaussianKernel(3,3)
image_smoothed = cv2.filter2D(image_resized, -1, K)

cv2.imshow('t', image_smoothed)
cv2.waitKey(0)