import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os


class XmlDocReader:
    images = []

    def __init__(self, file_id, path_images, xml_path):
        self.file_id = file_id
        self.path_images = path_images
        self.xml_path = xml_path

    def execute(self):
        id = 0
        self.images = []
        while True:
            image_full_path = os.path.join(self.path_images, '%s-%d.png' % (self.file_id, id))
            print(image_full_path)
            if not os.path.exists(image_full_path):
                break
            image = cv2.imread(image_full_path)
            self.images.append(image)
            id += 1
        self.see_doc()

    def see_region(self, region):
        page_number = int(region.attrib['page'])
        print("Checking doc", self.file_id, page_number)
        image = np.copy(self.images[page_number - 1])
        rows, cols, _ = np.shape(image)
        rows, cols = rows * 0.24, cols* 0.24
        for cell in region:
            bounding_box = cell.find('bounding-box')
            assert(bounding_box is not None)
            bounding_box = bounding_box.attrib
            x1 = int(bounding_box['x1'])
            y1 = rows - int(bounding_box['y1'])
            x2 = int(bounding_box['x2'])
            y2 = rows - int(bounding_box['y2'])
            cv2.rectangle(image, (int(x1 / 0.24), int(y1 / 0.24)), (int(x2 / 0.24), int(y2 / 0.24)), (0,0,255), thickness=2)
        image = cv2.resize(image, None, fx=0.25, fy=0.25)
        cv2.namedWindow('image')
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def see_table(self, table):
        for region in table:
            self.see_region(region)

    def see_doc(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        for table in root:
            self.see_table(table)


path = '/home/srq/Datasets/us-gov-dataset'
images_path = os.path.join(path, 'images')
for i in os.listdir(path):
    if not i.endswith('.pdf'):
        continue
    file_id = os.path.splitext(i)[0]
    full_pdf = os.path.join(path, i)
    full_str_xml = os.path.splitext(full_pdf)[0] + '-str.xml'
    reader = XmlDocReader(file_id, images_path, full_str_xml)
    reader.execute()