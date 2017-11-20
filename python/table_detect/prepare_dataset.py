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
from conv_autoencoder.convolutional_autoencoder import ConvolutionalAutoencoder
from table_detect.table_detect_document_2 import TableDetectDocument

show = False
show_ocr = False
dont_output = False

images_path = '/home/srq/Datasets/tables/unlv'
tables_gt_path = '/home/srq/Datasets/tables/unlv/unlv_xml_gt'
ocr_gt_path = '/home/srq/Datasets/tables/unlv/unlv_xml_ocr'

test_division_txt = '/home/srq/Datasets/tables/unlv-division/test.txt'
train_division_txt = '/home/srq/Datasets/tables/unlv-division/train.txt'
validate_division_txt = '/home/srq/Datasets/tables/unlv-division/validate.txt'

test_out = '/home/srq/Datasets/tables/unlv-for-detect/test'
train_out = '/home/srq/Datasets/tables/unlv-for-detect/train'
validate_out = '/home/srq/Datasets/tables/unlv-for-detect/validate'

glove_path = '/media/srq/Seagate Expansion Drive/Models/GloVe/glove.840B.300d.txt'
cache_name = 'unlv_complete'


class PrepareDataset:
    image = None

    def __init__(self, id, png_path, xml_path, ocr_path, sorted_path, glove_reader):
        self.id = id
        self.png_path = png_path
        self.xml_path = xml_path
        self.ocr_path = ocr_path
        self.sorted_path = sorted_path
        self.words_json = None
        self.glove_reader = glove_reader

    def execute(self):
        self.image = cv2.imread(self.png_path, 1)
        self.rows, self.cols, _ = np.shape(self.image)
        self.image_tables = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.conv_model = ConvolutionalAutoencoder()
        self.conv_model.prepare_for_manual_testing()
        self.see_ocr()
        self.see_doc()
        self.see_tokens()

    def see_ocr(self):
        image = np.copy(self.image)
        rows, cols, _ = np.shape(image)

        tree = ET.parse(self.ocr_path)
        root = tree.getroot()
        words_xml = root.find('words')
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

    def see_doc(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        tables = root.find('Tables')
        i = 1
        for table in tables:
            self.see_table(table, i)
            i += 1

    def see_table(self, table, increment):
        table_attributes = table.attrib
        tx1 = int(table_attributes['x0'])
        ty1 = int(table_attributes['y0'])
        tx2 = int(table_attributes['x1'])
        ty2 = int(table_attributes['y1'])

        self.image_tables[ty1:ty2 + 1, tx1:tx2 + 1] = increment

    def see_tokens(self):
        class_indices = []
        for i in range(len(self.all_tokens)):
            token = self.all_tokens[i]
            token_rect = self.all_tokens_rects[i]
        for i in range(len(self.all_tokens)):
            token_rect = self.all_tokens_rects[i]
            try:
                class_indices.append(0 if self.image_tables[int(token_rect['y'] + token_rect['height'] / 2), int(
                    token_rect['x'] + token_rect['width'] / 2)] == 0 else 1)
            except:
                print(i, self.all_tokens[i], self.all_tokens_rects[i])
                pass
        document_dump_path = os.path.join(self.sorted_path, '__dump__.pickle')
        spatial_features = self.conv_model.get_feature_map(self.image).astype(np.float64)
        self.dump_doc(self.all_tokens, self.all_tokens_rects, spatial_features, document_dump_path)

    def dump_doc(self, all_tokens, all_tokens_rects, spatial_features, file_name):
        N = len(all_tokens)
        height, width, _ = np.shape(self.image)
        classes = np.zeros(N)
        inside_same_table = np.zeros((N, 4))
        rect_matrix = np.zeros((N, 4))
        embeddings_matrix = np.zeros((N, 300))

        features_spatial_height, features_spatial_width, depth = np.shape(spatial_features)

        conv_features = np.zeros((N, depth))

        graph_builder = NeighborGraphBuilder(all_tokens_rects, self.image_tables)

        if not dont_output:
            if not os.path.exists(self.sorted_path):
                os.mkdir(self.sorted_path)

        neighbor_graph, neighbor_distance_matrix = graph_builder.get_neighbor_matrix()
        neighbor_distance_matrix[:, 0] = neighbor_distance_matrix[:, 0] / width
        neighbor_distance_matrix[:, 1] = neighbor_distance_matrix[:, 1] / height
        neighbor_distance_matrix[:, 2] = neighbor_distance_matrix[:, 2] / width
        neighbor_distance_matrix[:, 3] = neighbor_distance_matrix[:, 3] / height
        draw_image = np.copy(self.image)

        for i in range(N):
            token_rect = all_tokens_rects[i]
            index = self.image_tables[int(token_rect['y'] + token_rect['height'] / 2), int(
                token_rect['x'] + token_rect['width'] / 2)]

            left_rect = all_tokens_rects[int(neighbor_graph[i, 0])]
            top_rect = all_tokens_rects[int(neighbor_graph[i, 1])]
            right_rect = all_tokens_rects[int(neighbor_graph[i, 2])]
            bottom_rect = all_tokens_rects[int(neighbor_graph[i, 3])]

            if index == 0:
                index_left = index_right = index_top = index_bottom = 0
            else:
                index_left = 0 if self.image_tables[int(left_rect['y'] + left_rect['height'] / 2), int(
                    left_rect['x'] + left_rect['width'] / 2)] == index or int(neighbor_graph[i, 0]) == -1 else 1
                index_top = 0 if self.image_tables[int(top_rect['y'] + top_rect['height'] / 2), int(
                    top_rect['x'] + top_rect['width'] / 2)] == index or int(neighbor_graph[i, 1]) == -1 else 1
                index_right = 0 if self.image_tables[int(right_rect['y'] + right_rect['height'] / 2), int(
                    right_rect['x'] + right_rect['width'] / 2)] == index or int(neighbor_graph[i, 2]) == -1 else 1
                index_bottom = 0 if self.image_tables[int(bottom_rect['y'] + bottom_rect['height'] / 2), int(
                    bottom_rect['x'] + bottom_rect['width'] / 2)] == index or int(neighbor_graph[i, 3]) == -1 else 1

            inside_same_table[i, 0] = index_left
            inside_same_table[i, 1] = index_top
            inside_same_table[i, 2] = index_right
            inside_same_table[i, 3] = index_bottom

            color = (0, 0, 255) if index == 0 else (255, 0, 0)
            if index_left != 0 or index_top != 0 or index_right != 0 or index_bottom != 0:
                color = (0, 255, 0)
            cv2.rectangle(draw_image, (int(token_rect['x']), int(
                token_rect['y'])), (int(token_rect['x'] + token_rect['width']), int(
                token_rect['y'] + token_rect['height'])), color, 3)
        draw_path = os.path.join(self.sorted_path, 'visual.png')
        print(draw_path)
        cv2.imwrite(draw_path, draw_image)

        for i in range(N):
            token_rect = all_tokens_rects[i]
            index = 0 if self.image_tables[int(token_rect['y'] + token_rect['height'] / 2), int(
                token_rect['x'] + token_rect['width'] / 2)] == 0 else 1
            classes[i] = index
            rect_matrix[i, 0] = token_rect['x'] / width
            rect_matrix[i, 1] = token_rect['y'] / height
            rect_matrix[i, 2] = token_rect['width'] / width
            rect_matrix[i, 3] = token_rect['height'] / height

            feat_x = int((rect_matrix[i, 0] + rect_matrix[i, 2] / 2) * features_spatial_width)
            feat_y = int((rect_matrix[i, 1] + rect_matrix[i, 3] / 2) * features_spatial_height)

            assert feat_x < features_spatial_width and feat_y < features_spatial_height

            conv_features[i] = spatial_features[feat_y, feat_x]

            embedding = self.glove_reader.get_vector(all_tokens[i])
            if embedding is None:
                embedding = np.ones((300)) * (-1)
            embeddings_matrix[i] = embedding

        document = TableDetectDocument(embeddings_matrix, rect_matrix, neighbor_distance_matrix, neighbor_graph,
                                       classes, conv_features, inside_same_table)
        with open(file_name, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)


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

    PrepareDataset(id, png_path_full, xml_path_full, ocr_path_full, sorted_path_full, glove_reader).execute()
