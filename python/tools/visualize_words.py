import numpy as np
import cv2
import json
import sys
import os


if len(sys.argv) != 2:
    print("Error")
    sys.exit()

path = sys.argv[1]

for sub_folder in os.listdir(path):
    sub_folder_full = os.path.join(path, sub_folder)
    print(sub_folder_full)
    image_full = os.path.join(sub_folder_full, 'image.png')
    json_full = os.path.join(sub_folder_full, 'ocr_gt.json')
    visual_full = os.path.join(sub_folder_full, 'visual.png')
    image = cv2.imread(image_full)
    with open(json_full) as data_file:
        ocr_data = json.load(data_file)

    for i in range(len(ocr_data)):
        word_data = ocr_data[i]
        x, y, width, height = word_data['rect']['x'], word_data['rect']['y'], word_data['rect']['width'], \
                              word_data['rect']['height']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3)

    cv2.imwrite(visual_full, image)

#
#
# image_path = '/home/srq/Datasets/tables/unlv-for-nlp/train/1060_195/image.png'
# visual_path = '/home/srq/Datasets/tables/unlv-for-nlp/train/1060_195/visual.png'
# ocr_path = '/home/srq/Datasets/tables/unlv-for-nlp/train/1060_195/ocr.json'
#
# image = cv2.imread(image_path)
#
#
# with open(ocr_path) as data_file:
#     ocr_data = json.load(data_file)
#
# for i in range(len(ocr_data)):
#     word_data = ocr_data[i]
#     x, y, width, height = word_data['rect']['x'], word_data['rect']['y'], word_data['rect']['width'], word_data['rect']['height']
#     cv2.rectangle(image, (x,y), (x+width, y+height), (0,0,255), 3)
#
# cv2.imwrite(visual_path, image)