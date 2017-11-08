import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import json
import shutil
import tensorflow as tf
from object_detection.utils import dataset_util


image_path = '/home/srq/Datasets/fmarg/layout1/18197926/001.tif'
xml_path = '/home/srq/Datasets/fmarg/layout1/18197926/001.xml'


show_image = False

class MargReader:
    zone_x1s = []
    zone_y1s = []
    zone_x2s = []
    zone_y2s = []

    def __init__(self, image_path, xml_path, parent_dir_path):
        self.image_path = image_path
        self.xml_path = xml_path
        self.png_path = os.path.join(self.image_path, os.path.splitext(self.image_path)[0]+'.png')
        self.file_name = os.path.split(parent_dir_path)[-1]+'_'+os.path.split(self.png_path)[-1]

    def execute_zone(self, zone):
        lines = zone.findall('Line')
        print('\t%d lines' % len(lines))
        zone_corners = zone.find('ZoneCorners')
        vertices = zone_corners.findall('Vertex')
        x1 = int(vertices[0].attrib['x'])
        y1 = int(vertices[0].attrib['y'])
        x2 = int(vertices[2].attrib['x'])
        y2 = int(vertices[2].attrib['y'])
        self.zone_x1s.append(x1 / self.width)
        self.zone_y1s.append(y1 / self.height)
        self.zone_x2s.append(x2 / self.width)
        self.zone_y2s.append(y2 / self.height)
        print('\t%d %d %d %d' % (x1, y1, x2, y2))
        if show_image:
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

    def get_tf_example(self):
        height = self.height  # Image height
        width = self.width # Image width
        filename = self.file_name.encode()  # Filename of the image. Empty if image is not from file

        with open(self.png_path, 'rb') as f:
            all_bytes = f.read()

        encoded_image_data = all_bytes  # Encoded image bytes
        image_format = 'png'.encode()  # b'jpeg' or b'png'

        xmins = self.zone_x1s  # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = self.zone_x2s  # List of normalized right x coordinates in bounding box
        # (1 per box)
        ymins = self.zone_y1s  # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = self.zone_y2s  # List of normalized bottom y coordinates in bounding box
        # (1 per box)
        classes_text = ['zone'.encode()] * len(xmins)  # List of string class name of bounding box (1 per box)
        classes = [1] * len(xmins)  # List of integer class id of bounding box (1 per box)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def execute(self):
        self.image = cv2.imread(self.image_path)
        cv2.imwrite(self.png_path, self.image)
        self.height, self.width, _ = np.shape(self.image)
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        page = root.find('Page')
        zones = tree.findall('Zone')
        print(len(zones))
        for zone in zones:
            self.execute_zone(zone)

        if show_image:
            image = cv2.resize(self.image, None, fx=0.25, fy=0.25)
            cv2.namedWindow('result', cv2.WINDOW_FULLSCREEN)
            cv2.imshow('result', image)
            cv2.waitKey(0)

parent_path = '/home/srq/Datasets/fmarg/combined'
writer_train = tf.python_io.TFRecordWriter('train.record')
writer_validation = tf.python_io.TFRecordWriter('validation.record')
writer_test = tf.python_io.TFRecordWriter('test.record')

num = 0
for format_dir in os.listdir(parent_path):
    format_dir_full = os.path.join(parent_path, format_dir)
    if not os.path.isdir(format_dir_full):
        continue
    for image_file in os.listdir(format_dir_full):
        if not image_file.endswith('.tif'):
            continue
        image_file_full = os.path.join(format_dir_full, image_file)
        xml_file_full = os.path.join(format_dir_full, os.path.splitext(image_file)[0]+'.xml')
        print(image_file_full)
        print(xml_file_full,'\n')
        reader = MargReader(image_file_full, xml_file_full, format_dir_full)
        reader.execute()
        tf_example = reader.get_tf_example()

        if num < 900:
            writer_train.write(tf_example.SerializeToString())
            print("Writing training example")
        elif num < 1000:
            writer_validation.write(tf_example.SerializeToString())
            print("Writing validation example")
        else:
            writer_test.write(tf_example.SerializeToString())
            print("Writing test example")

        print('%.2f%% done' % (100 * num / 1405))

        num += 1

print("Number of files is ", num)

writer_train.close()
writer_validation.close()
writer_test.close()


# reader = MargReader(image_path, xml_path)
# reader.execute()
