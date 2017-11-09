import numpy as np
import cv2
import os
import sys


if len(sys.argv) != 2:
    print("Error")
    sys.exit()

path = sys.argv[1]

exe_path = '/home/srq/Projects/TIES/cpp/cmake-build-debug/tessocr'

for sub_folder in os.listdir(path):
    sub_folder_full = os.path.join(path, sub_folder)
    image_full = os.path.join(sub_folder_full, 'image.png')
    print(image_full)
    json_full = os.path.join(sub_folder_full, 'ocr.json')
    os.system('%s %s %s' % (exe_path, image_full, json_full))
