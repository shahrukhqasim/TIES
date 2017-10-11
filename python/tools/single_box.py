import cv2
import numpy as np


image_path = '/home/srq/Datasets/us-gov-dataset/output.png'
image = cv2.imread(image_path)

x1 = 489
x2 = 504
y1 = 673
y2 = 681


W = 612#595
H = 792#842

y1 = H - y1
y2 = H - y2

print(y1, y2)

cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 1)

cv2.imshow('output', image)
cv2.waitKey(0)