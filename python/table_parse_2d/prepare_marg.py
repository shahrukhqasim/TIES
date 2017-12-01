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
from table_parse.table_data import TableData
from conv_autoencoder.convolutional_autoencoder import ConvolutionalAutoencoder
import math
from table_parse_2d.document_for_zone_segment import ZoneSegmentDocument
import gzip

show = False
show_ocr = False
dont_output = False

input_path = '/home/srq/Datasets/fmarg/combined'

test_division_txt = '/home/srq/Datasets/fmarg/division/test.txt'
train_division_txt = '/home/srq/Datasets/fmarg/division/train.txt'
validate_division_txt = '/home/srq/Datasets/fmarg/division/validate.txt'

test_out = '/home/srq/Datasets/fmarg/marg-for-div/test'
train_out = '/home/srq/Datasets/fmarg/marg-for-div/train'
validate_out = '/home/srq/Datasets/fmarg/marg-for-div/validate'

glove_path = '/media/srq/Seagate Expansion Drive1/Models/GloVe/glove.840B.300d.txt'
cache_name = 'marg_complete'


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


class PrepareMarg:
    def __init__(self, png_path, xml_path, ocr_json_path, sorted_path_parent, glove_reader):
        self.png_path = png_path
        self.xml_path = xml_path
        self.sorted_path = sorted_path_parent
        self.ocr_json_path = ocr_json_path
        self.glove_reader = glove_reader

    def execute_zone(self, zone, zone_id):
        zone_corners = zone.find('ZoneCorners')
        vertices = zone_corners.findall('Vertex')
        x1 = int(vertices[0].attrib['x'])
        y1 = int(vertices[0].attrib['y'])
        x2 = int(vertices[2].attrib['x'])
        y2 = int(vertices[2].attrib['y'])

        self.zone_segmentation[y1:y2-1, x1:x2-1] = zone_id

    def execute_tokens(self):
        # To get local neighbors of each token: Left, right, top, bottom
        graph_builder = NeighborGraphBuilder(self.all_tokens_rects, self.image[:,:,0])
        # M is the indices graph and D is distance matrix
        M, D = graph_builder.get_neighbor_matrix()

        N = len(self.all_tokens)

        neighbors_same_zone = np.zeros((N,4))

        for i in range(N):
            left_index = int(M[i,0])
            top_index = int(M[i,1])
            right_index = int(M[i,2])
            bottom_index = int(M[i,3])

            token_rect = self.all_tokens_rects[i]
            mid = [int(token_rect['x'] + token_rect['width'] / 2), int(token_rect['y'] + token_rect['height'] / 2)]

            if left_index != -1:
                token_rect_2 = self.all_tokens_rects[left_index]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share zone
                if self.zone_segmentation[mid[1], mid[0]] == self.zone_segmentation[mid_2[1], mid_2[0]]:
                    neighbors_same_zone[i, 0] = 1

            if top_index != -1:
                token_rect_2 = self.all_tokens_rects[top_index]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share zone
                if self.zone_segmentation[mid[1], mid[0]] == self.zone_segmentation[mid_2[1], mid_2[0]]:
                    neighbors_same_zone[i, 1] = 1

            if right_index != -1:
                token_rect_2 = self.all_tokens_rects[right_index]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share zone
                if self.zone_segmentation[mid[1], mid[0]] == self.zone_segmentation[mid_2[1], mid_2[0]]:
                    neighbors_same_zone[i, 2] = 1

            if bottom_index != -1:
                token_rect_2 = self.all_tokens_rects[bottom_index]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share zone
                if self.zone_segmentation[mid[1], mid[0]] == self.zone_segmentation[mid_2[1], mid_2[0]]:
                    neighbors_same_zone[i, 3] = 1

        # To place input vectors at respective spatial coordinates
        input_tensor = np.zeros((256, 256, 308)).astype(np.float64)
        # Same zone or not, 0 for not, 1 for yes
        output_tensor = np.zeros((256, 256, 4)).astype(np.float64)
        # Whether there was a word here or not
        output_tensor_word_mask = np.zeros((256, 256)).astype(np.float64)
        # Whether there was a zone here or not
        self.zone_segmentation[self.zone_segmentation != 0] = 1
        output_tensor_zone_mask = cv2.resize(self.zone_segmentation, (256,256))
        for i in range(N):
            token_rect = self.all_tokens_rects[i]
            # Source coordinates of top left of tokens
            cx = token_rect['x']
            cy = token_rect['y']
            cw = token_rect['width']
            ch = token_rect['height']


            distances_vector = D[i]

            # Get the GloVe reading
            embedding = self.glove_reader.get_vector(self.all_tokens[i])
            if embedding is None:
                embedding = np.ones((300)) * (-1)

            positional = np.array([cx / self.width, cx / self.height, cw / self.width, ch / self.width,
                                   distances_vector[0] / self.width, distances_vector[1] / self.height,
                                   distances_vector[2] / self.width, distances_vector[3] / self.height])

            # Destination coordinates on 256x256 scale and place there
            nx = math.floor(256.0 * cx / self.width)
            ny = math.floor(256.0 * cy / self.height)
            input_tensor[ny, nx] = np.concatenate((embedding, positional))

            # From the neighbor graph
            output_tensor[ny, nx] = np.array([neighbors_same_zone[i, 0], neighbors_same_zone[i, 1], neighbors_same_zone[i, 2],
                                              neighbors_same_zone[i, 3]])
            # Set mask to 1
            output_tensor_word_mask[ny, nx] = 1

        print(self.sorted_path)

        # Output debugging visual file for zone mask
        segmentation_visualize_path = os.path.join(self.sorted_path, 'visual_segment.png')
        cv2.imwrite(segmentation_visualize_path, (output_tensor_zone_mask*255).astype(np.uint8))

        # Output debugging visual image for word mask
        word_mask_path = os.path.join(self.sorted_path, 'visual_word_mask.png')
        cv2.imwrite(word_mask_path, (output_tensor_word_mask * 255).astype(np.uint8))

        # Dump the content to pickle file. The file is compressed by gzip.
        dump_path = os.path.join(self.sorted_path, '__dump__.pklz')
        document = ZoneSegmentDocument(input_tensor, output_tensor, output_tensor_word_mask, output_tensor_zone_mask)
        f = gzip.open(dump_path, 'wb')
        pickle.dump(document, f)
        f.close()

    def execute(self):
        self.image = cv2.imread(self.png_path)
        self.height, self.width, _ = np.shape(self.image)
        self.zone_segmentation = np.zeros((self.height, self.width)).astype(np.uint8)
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        page = root.find('Page')
        zones = tree.findall('Zone')
        zone_id = 1
        for zone in zones:
            self.execute_zone(zone, zone_id)
            zone_id += 1
        self.see_words()
        self.execute_tokens()

    def see_words(self):
        ocr_data, _ = PrepareMarg.pick_up_words(self.ocr_json_path, self.png_path)
        self.all_tokens_rects = []
        self.all_tokens = []
        for word in ocr_data:
            x1 = int(word['rect']['x'])
            y1 = int(word['rect']['y'])
            x2 = x1 + int(word['rect']['width'])
            y2 = y1 + int(word['rect']['height'])
            word_text = word['word']

            tokens = nltk.word_tokenize(word_text)
            word_rect = {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}
            divided_width = word_rect['width'] / len(tokens)
            self.all_tokens.extend(tokens)
            for j in range(len(tokens)):
                token_rect = dict(word_rect)
                token_rect['x'] += int(j * divided_width)
                token_rect['width'] = int(divided_width)
                self.all_tokens_rects.append(token_rect)




    @staticmethod
    def pick_up_words(json_path, image_path):
        image = cv2.imread(image_path, 0)
        height, width = np.shape(image)

        with open(json_path) as data_file:
            ocr_data = json.load(data_file)

        ocr_data_2 = []
        nlp_tokens_all = set()
        for i in range(len(ocr_data)):

            word_data = ocr_data[i]
            x, y, width, height, word = int(word_data['rect']['x']), int(word_data['rect']['y']), int(word_data['rect']['width']), \
                                   int(word_data['rect']['height']), word_data['word']

            word_data_2 = {'rect': {'x': x, 'y': y, 'width': width, 'height': height}, 'word': word}

            tokens = nltk.word_tokenize(word)
            for j in tokens:
                nlp_tokens_all.add(j)

            ocr_data_2.append(word_data_2)

        return ocr_data_2, nlp_tokens_all



# print("Loading dictionary")
nlp_tokens = set()
# i = 0
# for parent_path in os.listdir(input_path):
#     parent_path_full = os.path.join(input_path, parent_path)
#
#     for sub_file in os.listdir(parent_path_full):
#         if not sub_file.endswith('.png'):
#             continue
#         id = os.path.splitext(sub_file)[0]
#         png_path = os.path.join(parent_path_full, id+'.png')
#         json_path = os.path.join(parent_path_full, id+'.json')
#         _, new_tokens = PrepareMarg.pick_up_words(json_path, png_path)
#         nlp_tokens = nlp_tokens.union(new_tokens)
#         print(id, i)
#         i += 1
#
# print("Found",len(nlp_tokens),"unique tokens")

glove_reader = GLoVe(glove_path, nlp_tokens)
glove_reader.load(cache_name)


last_id = 1

for parent_path in os.listdir(input_path):
    parent_path_full = os.path.join(input_path, parent_path)

    if parent_path in test_ids:
        out_path = test_out
    elif parent_path in validate_ids:
        out_path = validate_out
    elif parent_path in train_ids:
        out_path = train_out
    else:
        assert False

    for sub_file in os.listdir(parent_path_full):
        if not sub_file.endswith('.png'):
            continue
        id = os.path.splitext(sub_file)[0]
        png_path = os.path.join(parent_path_full, id+'.png')
        xml_path = os.path.join(parent_path_full, id+'.xml')
        json_path = os.path.join(parent_path_full, id+'.json')
        sorted_path = os.path.join(out_path, str(last_id))
        if not os.path.exists(sorted_path):
            os.mkdir(sorted_path)
        assert os.path.exists(png_path) and os.path.exists(xml_path) and os.path.exists(json_path)

        PrepareMarg(png_path, xml_path, json_path, sorted_path, glove_reader).execute()

        last_id += 1
