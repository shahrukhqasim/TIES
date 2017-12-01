import numpy as np
import json
import xml.etree.ElementTree as ET
import cv2
import os

input_path = '/home/srq/Datasets/fmarg/combined'
tesseract_path = '/home/srq/Projects/TIES/cpp/cmake-build-debug/tessocr'

for parent_path in os.listdir(input_path):
    parent_path_full = os.path.join(input_path, parent_path)

    for sub_file in os.listdir(parent_path_full):
        if not sub_file.endswith('.png'):
            continue
        print(sub_file)
        id = os.path.splitext(sub_file)[0]
        png_path = os.path.join(parent_path_full, id+'.png')
        json_out_path = os.path.join(parent_path_full, id+'.json')
        assert os.path.exists(png_path)
        command = '%s %s %s' % (tesseract_path, png_path, json_out_path)
        os.system(command)