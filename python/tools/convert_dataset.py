import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import json
import shutil

show = True
show_ocr = False
dont_output = True


class UnlvConverter:
    image = None
    table_segment = None

    def __init__(self, id, png_path, xml_path, ocr_path, sorted_path):
        self.id = id
        self.png_path = png_path
        self.xml_path = xml_path
        self.ocr_path = ocr_path
        self.sorted_path = sorted_path
        self.words_json = None

    def execute(self):
        self.image = cv2.imread(self.png_path)
        height, width, _ = np.shape(self.image)
        self.table_segment = np.zeros((height, width), dtype=np.uint8)
        # self.see_ocr()
        self.see_doc()

    def see_table(self, table, increment):
        print("Converting doc", self.png_path)

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

        cv2.rectangle(self.table_segment, (int(x1),int(y1)), (int(x2), int(y2)), 255, cv2.FILLED)

    def see_doc(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        tables = root.find('Tables')
        i = 0
        for table in tables:
            self.see_table(table, i)
            i += 1

        os.mkdir(self.sorted_path)
        shutil.copy(self.png_path, os.path.join(self.sorted_path, 'image.png'))
        cv2.imwrite(os.path.join(self.sorted_path, 'tables.png'), self.table_segment)


    def see_ocr(self):
        image = np.copy(self.image)
        rows, cols, _ = np.shape(image)

        tree = ET.parse(self.ocr_path)
        root = tree.getroot()
        words_xml = root.find('words')
        words = []
        for word_xml in words_xml:
            word_text = word_xml.text
            word_xml_attrib = word_xml.attrib
            x1 = int(word_xml_attrib['left'])
            y1 = rows - int(word_xml_attrib['top'])
            x2 = int(word_xml_attrib['right'])
            y2 = rows - int(word_xml_attrib['bottom'])

            if y1 > y2:
                y1, y2 = y2, y1

            if show_ocr:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),
                              thickness=2)

            word = dict()
            word['text'] = word_text
            word['x1'] = x1
            word['y1'] = y1
            word['x2'] = x2
            word['y2'] = y2
            words.append(word)
        json_out = dict()
        json_out['words'] = words
        self.words_json = json_out



gt_path = '/home/srq/Datasets/tables/unlv/unlv_xml_gt'
images_path = '/home/srq/Datasets/tables/unlv'
ocr_path = '/home/srq/Datasets/tables/unlv/unlv_xml_ocr'
sorted_path = '/home/srq/Datasets/tables/unlv-for-segment'

for i in os.listdir(gt_path):
    if not i.endswith('.xml'):
        continue
    file_id = os.path.splitext(i)[0]
    xml_full_path = os.path.join(gt_path,i)
    png_full_path = os.path.join(images_path, file_id + '.png')
    ocr_full_path = os.path.join(ocr_path, i)
    reader = UnlvConverter(file_id, png_full_path, xml_full_path, ocr_full_path, os.path.join(sorted_path, file_id))
    reader.execute()