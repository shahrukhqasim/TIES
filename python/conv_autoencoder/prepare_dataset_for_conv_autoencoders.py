import torch
import numpy as np
import cv2
import os
import shutil

images_path = '/home/srq/Datasets/tables/uw3/correctgt-zones/'
output_path = '/home/srq/Datasets/tables/uw3-for-auto-encoders'

for i in os.listdir(images_path):
    if not i.endswith('.png'):
        continue
    id = os.path.splitext(i)[0]
    print(id)
    output_create_dir = os.path.join(output_path,id)
    if not os.path.exists(output_create_dir):
        os.mkdir(output_create_dir)
    image_out = os.path.join(output_create_dir, 'image.png')
    shutil.copy(os.path.join(images_path, i), image_out)

