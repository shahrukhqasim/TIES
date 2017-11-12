import numpy as np
import json
import xml.etree.ElementTree as ET
import cv2
import os
import nltk
import shutil
from network.neighbor_graph_builder import NeighborGraphBuilder
from network.glove_reader import GLoVe
import pickle
from network.table_data import TableData

show = False
show_ocr = False
dont_output = False

images_path = '/home/srq/Datasets/tables/unlv'
tables_gt_path = '/home/srq/Datasets/tables/unlv/unlv_xml_gt'
ocr_gt_path = '/home/srq/Datasets/tables/unlv/unlv_xml_ocr'

test_division_txt = '/home/srq/Datasets/tables/unlv-division/test.txt'
train_division_txt = '/home/srq/Datasets/tables/unlv-division/train.txt'
validate_division_txt = '/home/srq/Datasets/tables/unlv-division/validate.txt'

test_out = '/home/srq/Datasets/tables/unlv-for-table-parsing/test'
train_out = '/home/srq/Datasets/tables/unlv-for-table-parsing/train'
validate_out = '/home/srq/Datasets/tables/unlv-for-table-parsing/validate'

glove_path = '/media/srq/Seagate Expansion Drive/Models/GloVe/glove.840B.300d.txt'
cache_name = 'unlv_complete'


class UnlvConverter:
    image = None

    def __init__(self, id, png_path, xml_path, ocr_path, sorted_path):
        self.id = id
        self.png_path = png_path
        self.xml_path = xml_path
        self.ocr_path = ocr_path
        self.sorted_path = sorted_path
        self.words_json = None

    def execute(self):
        self.image = cv2.imread(self.png_path, 1)
        self.rows, self.cols, _ = np.shape(self.image)
        self.see_ocr()
        self.see_doc()

    def see_table(self, table, increment):
        print("Converting doc", self.png_path)

        table_attributes = table.attrib
        tx1 = int(table_attributes['x0'])
        ty1 = int(table_attributes['y0'])
        tx2 = int(table_attributes['x1'])
        ty2 = int(table_attributes['y1'])

        sorted_path_full = self.sorted_path + "-%d" % increment
        if not dont_output:
            if not os.path.exists(sorted_path_full):
                os.mkdir(sorted_path_full)

        data_image = np.zeros((self.rows, self.cols, 3), dtype=np.int32)

        rows_xml = table.findall('Row')
        rows_matrix = np.zeros((len(rows_xml), 4))
        rr = 0
        last_y = ty1
        for row in rows_xml:
            row_attrib = row.attrib
            x1 = rows_matrix[rr, 0] = int(row_attrib['x0'])
            y1 = rows_matrix[rr, 1] = int(row_attrib['y0'])
            x2 = rows_matrix[rr, 2] = int(row_attrib['x1'])
            y2 = rows_matrix[rr, 3] = int(row_attrib['y1'])
            rr += 1
            data_image[last_y: y1 + 1, x1:x2 + 1, 0] = rr
            last_y = y1 + 1
        data_image[last_y: ty2, tx1:tx2 + 1, 0] = rr

        columns_xml = table.findall('Column')
        cols_matrix = np.zeros((len(columns_xml), 4))
        cc = 0
        last_x = tx1
        for col in columns_xml:
            col_attrib = col.attrib
            x1 = cols_matrix[cc, 0] = int(col_attrib['x0'])
            y1 = cols_matrix[cc, 1] = int(col_attrib['y0'])
            x2 = cols_matrix[cc, 2] = int(col_attrib['x1'])
            y2 = cols_matrix[cc, 3] = int(col_attrib['y1'])
            cc += 1
            data_image[y1:y2 + 1, last_x:x1 + 1, 1] = cc
            last_x = x1 + 1
        data_image[ty1:ty2, last_x:tx2, 1] = cc

        cells_xml = table.findall('Cell')
        ll = 0
        for cell_xml in cells_xml:
            bounding_box = cell_xml.attrib
            if bounding_box['dontCare'] == 'true':
                continue
            x1 = int(bounding_box['x0'])
            y1 = int(bounding_box['y0'])
            x2 = int(bounding_box['x1'])
            y2 = int(bounding_box['y1'])
            ll += 1
            data_image[y1:y2 + 1, x1:x2 + 1, 2] = ll
        show_1 = ((data_image[:, :] * 100) % 256).astype(np.uint8)
        if show:
            # show_2 = ((data_image[:,:,1] * 100) % 256).astype(np.uint8)
            # show_3 = ((data_image[:,:,2] * 100) % 256).astype(np.uint8)

            show_1 = cv2.resize(show_1, None, fx=0.25, fy=0.25)
            cv2.imshow('rows', show_1)
            # show_2 = cv2.resize(show_2, None, fx=0.25, fy=0.25)
            # cv2.imshow('cols', show_2)
            # show_3 = cv2.resize(show_3, None, fx=0.25, fy=0.25)
            # cv2.imshow('cells', show_3)

            cv2.waitKey(0)

        all_tokens = []
        all_tokens_rects = []
        for i in range(len(self.all_tokens)):
            token = self.all_tokens[i]
            token_rect = self.all_tokens_rects[i]
            mid = [int(token_rect['x'] + token_rect['width'] / 2), int(token_rect['y'] + token_rect['height'] / 2)]
            if data_image[mid[1], mid[0], 0] == 0:
                continue
            all_tokens.append(token)
            all_tokens_rects.append(token_rect)

        N = len(all_tokens)

        row_share_matrix = np.zeros((N, N))
        col_share_matrix = np.zeros((N, N))
        cell_share_matrix = np.zeros((N, N))

        graph_builder = NeighborGraphBuilder(all_tokens_rects, data_image[:,:,0])
        M, D = graph_builder.get_neighbor_matrix()

        for i in range(N):
            token = all_tokens[i]
            token_rect = all_tokens_rects[i]
            mid = [int(token_rect['x'] + token_rect['width'] / 2), int(token_rect['y'] + token_rect['height'] / 2)]
            for j in range(N):
                token_2 = all_tokens[j]
                token_rect_2 = all_tokens_rects[j]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share row
                if data_image[mid[1], mid[0], 0] == data_image[mid_2[1], mid_2[0], 0]:
                    row_share_matrix[i, j] = 1
                # They share column
                if data_image[mid[1], mid[0], 1] == data_image[mid_2[1], mid_2[0], 1]:
                    col_share_matrix[i, j] = 1
                # They share cell
                if data_image[mid[1], mid[0], 2] == data_image[mid_2[1], mid_2[0], 2]:
                    cell_share_matrix[i, j] = 1

        self.dump_table(all_tokens, all_tokens_rects, M, D, row_share_matrix, col_share_matrix, cell_share_matrix, show_1, os.path.join(sorted_path_full, '__dump__.pickle'))
        cv2.imwrite(os.path.join(sorted_path_full, 'visual.png'), show_1)

    def dump_table(self, all_tokens, all_tokens_rects, neighbor_graph, neighbor_distance_matrix, share_row_matrix,
                   share_col_matrix, share_cell_matrix, image_visual, file_name):
        N = len(all_tokens)
        height, width, _ = np.shape(image_visual)
        classes = np.zeros(N)
        rect_matrix = np.zeros((N, 4))
        embeddings_matrix = np.zeros((N, 300))
        for i in range(N):
            token_rect = all_tokens_rects[i]
            rect_matrix[i, 0] = token_rect['x'] / width
            rect_matrix[i, 1] = token_rect['y'] / height
            rect_matrix[i, 2] = token_rect['width'] / width
            rect_matrix[i, 3] = token_rect['height'] / height
            embedding = glove_reader.get_vector(all_tokens[i])
            if embedding is None:
                embedding = np.ones((300)) * (-1)
            embeddings_matrix[i] = embedding

        document = TableData(embeddings_matrix, rect_matrix, neighbor_distance_matrix, neighbor_graph, share_row_matrix, share_col_matrix, share_cell_matrix)
        with open(file_name, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

    def see_doc(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        tables = root.find('Tables')
        i = 0
        for table in tables:
            self.see_table(table, i)
            i += 1

    def see_ocr(self):
        image = np.copy(self.image)
        rows, cols, _ = np.shape(image)

        tree = ET.parse(self.ocr_path)
        root = tree.getroot()
        words_xml = root.find('words')
        words = []
        self.all_tokens_rects = []
        self.all_tokens = []
        for word_xml in words_xml:
            word_text = word_xml.text
            word_xml_attrib = word_xml.attrib
            x1 = int(word_xml_attrib['left'])
            y1 = rows - int(word_xml_attrib['top'])
            x2 = int(word_xml_attrib['right'])
            y2 = rows - int(word_xml_attrib['bottom'])

            if y1 > y2:
                y1, y2 = y2, y1
            tokens = nltk.word_tokenize(word_text)
            word_rect = {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
            divided_width = word_rect['width'] / len(tokens)
            self.all_tokens.extend(tokens)
            for j in range(len(tokens)):
                token_rect = dict(word_rect)
                token_rect['x'] += int(j * divided_width)
                token_rect['width'] = int(divided_width)
                self.all_tokens_rects.append(token_rect)


def pick_up_words(json_path, image_path):
    image = cv2.imread(image_path, 0)
    height, width = np.shape(image)

    with open(json_path) as data_file:
        ocr_data = json.load(data_file)

    ocr_data_2 = []
    nlp_tokens_all = set()
    for i in range(len(ocr_data)):

        word_data = ocr_data[i]
        x1, y1, x2, y2, word = int(word_data['left']), int(word_data['top']), int(word_data['right']), \
                               int(word_data['bottom']), word_data['word']
        y1 = height - y1
        y2 = height - y2
        word_data_2 = {'rect': {'x': x1, 'y': y1, 'width': (x2 - x1), 'height': (y2 - y1)}, 'word': word}

        tokens = nltk.word_tokenize(word)
        for j in tokens:
            nlp_tokens_all.add(j)

        ocr_data_2.append(word_data_2)

    return ocr_data_2, nlp_tokens_all


# Pick up train/test/validate split
with open(train_division_txt, 'r') as f:
    train_ids = f.readlines()
    train_ids = [i[:-1] for i in train_ids]

with open(validate_division_txt, 'r') as f:
    validate_ids = f.readlines()
    validate_ids = [i[:-1] for i in validate_ids]

with open(test_division_txt, 'r') as f:
    test_ids = f.readlines()
    test_ids = [i[:-1] for i in test_ids]
print("Train set contains", len(train_ids))
print("Validate set contains", len(validate_ids))
print("Test set contains", len(test_ids))

# Pick up all tokens from NLTK
ii = 0
all_tokens = set()
# for i in os.listdir(images_path):
#     if not i.endswith('.png'):
#         continue
#     id = os.path.splitext(i)[0]
#     image_path_full = os.path.join(images_path, i)
#     ocr_path_full = os.path.join(ocr_gt_path, id+'.json')
#     _, all_tokens_doc = pick_up_words(ocr_path_full, image_path_full)
#     all_tokens = all_tokens.union(all_tokens_doc)
#     ii += 1
#     print("Phase 0, done: ", ii)
#
# print("Picked %d tokens from the dataset" % len(all_tokens))

glove_reader = GLoVe(glove_path, all_tokens)
glove_reader.load(cache_name)


for i in os.listdir(images_path):
    if not i.endswith('.png'):
        continue
    id = os.path.splitext(i)[0]
    png_path_full = os.path.join(images_path, i)
    xml_path_full = os.path.join(tables_gt_path, id + '.xml')
    ocr_path_full = os.path.join(ocr_gt_path, id + '.xml')
    sorted_path = None
    if id in train_ids:
        sorted_path = train_out
    elif id in validate_ids:
        sorted_path = validate_out
    elif id in test_ids:
        sorted_path = test_out
    sorted_path_full = os.path.join(sorted_path, i)

    UnlvConverter(id, png_path_full, xml_path_full, ocr_path_full, sorted_path_full).execute()
