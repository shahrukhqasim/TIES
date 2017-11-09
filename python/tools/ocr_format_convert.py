import cv2
import json
import os
import sys
import numpy as np


if len(sys.argv) != 3:
    print("Error")
    sys.exit()


set_path = sys.argv[1]
ocr_gt_path = sys.argv[2]



for sub_folder in os.listdir(set_path):
    sub_folder_full = os.path.join(set_path, sub_folder)
    json_out_path = os.path.join(sub_folder_full, 'ocr_gt.json')
    print(json_out_path)
    json_gt_path = os.path.join(ocr_gt_path, sub_folder+'.json')
    image_full = os.path.join(sub_folder_full, 'image.png')
    image = cv2.imread(image_full)
    height, width, _ = np.shape(image)

    with open(json_gt_path) as data_file:
        ocr_data = json.load(data_file)

    ocr_data_2 = []
    for i in range(len(ocr_data)):

        word_data = ocr_data[i]
        x1, y1, x2, y2, word = int(word_data['left']), int(word_data['top']), int(word_data['right']), \
                              int(word_data['bottom']), word_data['word']
        y1 = height - y1
        y2 = height - y2
        word_data_2 = {'rect' : {'x' : x1, 'y': y1, 'width' : (x2-x1), 'height' : (y2-y1)}, 'word' : word}
        ocr_data_2.append(word_data_2)

    with open(json_out_path, 'w') as out_file:
        json.dump(ocr_data_2, out_file)