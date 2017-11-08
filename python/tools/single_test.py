import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os


image_path = '/home/srq/Datasets/tables/unlv/9514_049.png'
xml_path = '/home/srq/Datasets/tables/unlv/unlv_xml_gt/9514_049.xml'

image_global = cv2.imread(image_path, 1)


def see_table(table, increment):
    print("Converting doc", image_path)

    table_attributes = table.attrib
    x1 = table_attributes['x0']
    y1 = table_attributes['y0']
    x2 = table_attributes['x1']
    y2 = table_attributes['y1']
    table_json = dict()
    table_json['table'] = dict()
    table_json['table']['x1'] = x1
    table_json['table']['y1'] = y1
    table_json['table']['x2'] = x2
    table_json['table']['y2'] = y2

    image = np.copy(image_global)
    image = cv2.resize(image, None, fx=0.25, fy=0.25)
    rows, cols, _ = np.shape(image)
    rows, cols = rows * 1, cols * 1
    cells_xml = table.findall('Cell')
    cells = []
    for cell_xml in cells_xml:
        bounding_box = cell_xml.attrib
        dc = bounding_box['dontCare']
        if dc == 'true':
            continue
        x1 = int(bounding_box['x0'])
        y1 = int(bounding_box['y0'])
        x2 = int(bounding_box['x1'])
        y2 = int(bounding_box['y1'])
        if True:
            image = np.copy(image_global)
            cv2.rectangle(image, (int(x1 / 1), int(y1 / 1)), (int(x2 / 1), int(y2 / 1)), (0, 0, 255), thickness=2)
            image = cv2.resize(image, None, fx=0.25, fy=0.25)
            cv2.namedWindow('image')
            cv2.imshow('image', image)
            cv2.waitKey(0)
        cell = dict()
        cell['x1'] = x1
        cell['y1'] = y1
        cell['x2'] = x2
        cell['y2'] = y2
        cells.append(cell)
    json_out = dict()
    json_out['cells'] = cells


def see_doc():
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tables = root.find('Tables')
    i = 0
    for table in tables:
        see_table(table, i)
        i += 1

see_doc()