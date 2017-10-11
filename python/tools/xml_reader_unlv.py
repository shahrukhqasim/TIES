import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os


class XmlDocReader:
    image = None

    def __init__(self, xml_path, png_path):
        self.xml_path = xml_path
        self.png_path = png_path

    def execute(self):
        self.image = cv2.imread(self.png_path)
        self.see_doc()

    def see_table(self, table):
        print("Checking doc", self.png_path)
        image = np.copy(self.image)
        rows, cols, _ = np.shape(image)
        rows, cols = rows * 1, cols* 1
        cells = table.findall('Cell')
        for cell in cells:
            bounding_box = cell.attrib
            x1 = int(bounding_box['x0'])
            y1 = int(bounding_box['y0'])
            x2 = int(bounding_box['x1'])
            y2 = int(bounding_box['y1'])
            cv2.rectangle(image, (int(x1 / 1), int(y1 / 1)), (int(x2 / 1), int(y2 / 1)), (0,0,255), thickness=2)
        image = cv2.resize(image, None, fx=0.25, fy=0.25)
        cv2.namedWindow('image')
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def see_doc(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        tables = root.find('Tables')
        for table in tables:
            self.see_table(table)


gt_path = '/home/srq/Datasets/unlv/unlv_xml_gt'
images_path = '/home/srq/Datasets/unlv'
for i in os.listdir(gt_path):
    if not i.endswith('.xml'):
        continue
    file_id = os.path.splitext(i)[0]
    xml_full_path = os.path.join(gt_path,i)
    png_full_path = os.path.join(images_path, file_id + '.png')
    reader = XmlDocReader(xml_full_path, png_full_path)
    reader.execute()