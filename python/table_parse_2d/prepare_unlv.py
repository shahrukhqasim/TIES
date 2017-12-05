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
import configparser as cp
import matplotlib
import gzip
import math
from table_parse_2d.document_for_table_parse import TableParseDocument

show = False
show_ocr = False
dont_output = False


config = cp.ConfigParser()
config.read('config.ini')

images_path=config['dataset_prepare_unlv']['images_path']
tables_gt_path=config['dataset_prepare_unlv']['tables_gt_path']
ocr_gt_path=config['dataset_prepare_unlv']['ocr_gt_path']
test_division_txt=config['dataset_prepare_unlv']['test_division_txt']
train_division_txt=config['dataset_prepare_unlv']['train_division_txt']
validate_division_txt=config['dataset_prepare_unlv']['validate_division_txt']
test_out=config['dataset_prepare_unlv']['test_out']
train_out=config['dataset_prepare_unlv']['train_out']
validate_out=config['dataset_prepare_unlv']['validate_out']
glove_path=config['dataset_prepare_unlv']['glove_path']
cache_name=config['dataset_prepare_unlv']['cache_name']


class UnlvConverter:
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
        self.height, self.width = self.rows, self.cols
        self.see_ocr()
        self.see_doc()

    def see_table(self, table, increment):
        print("Converting doc", self.png_path)

        table_attributes = table.attrib
        tx1 = int(table_attributes['x0'])
        ty1 = int(table_attributes['y0'])
        tx2 = int(table_attributes['x1'])
        ty2 = int(table_attributes['y1'])

        image_table_cropped = self.image[ty1:ty2+1, tx1:tx2+1]

        # _, _, 0 = row share
        # _, _, 1 = column share
        # _, _, 2 = cell share
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

            # show_1 = cv2.resize(show_1, None, fx=0.25, fy=0.25)
            # cv2.imshow('rows', show_1)
            # # show_2 = cv2.resize(show_2, None, fx=0.25, fy=0.25)
            # # cv2.imshow('cols', show_2)
            # # show_3 = cv2.resize(show_3, None, fx=0.25, fy=0.25)
            # # cv2.imshow('cells', show_3)
            #
            # cv2.waitKey(0)
            pass

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

        if N == 0:
            return # If there are no words in the table, its useless anyway

        row_share_matrix = np.zeros((N, N))
        col_share_matrix = np.zeros((N, N))
        cell_share_matrix = np.zeros((N, N))

        neighbors_same_row = np.zeros((N,4))
        neighbors_same_col = np.zeros((N,4))
        neighbors_same_cell = np.zeros((N,4))

        graph_builder = NeighborGraphBuilder(all_tokens_rects, data_image[:,:,0])
        M, D = graph_builder.get_neighbor_matrix()

        for i in range(N):
            left_index = int(M[i,0])
            top_index = int(M[i,1])
            right_index = int(M[i,2])
            bottom_index = int(M[i,3])

            token_rect = all_tokens_rects[i]
            mid = [int(token_rect['x'] + token_rect['width'] / 2), int(token_rect['y'] + token_rect['height'] / 2)]

            if left_index != -1:
                token_rect_2 = all_tokens_rects[left_index]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share row
                if data_image[mid[1], mid[0], 0] == data_image[mid_2[1], mid_2[0], 0]:
                    neighbors_same_row[i, 0] = 1
                # They share column
                if data_image[mid[1], mid[0], 1] == data_image[mid_2[1], mid_2[0], 1]:
                    neighbors_same_col[i, 0] = 1
                # They share cell
                if data_image[mid[1], mid[0], 2] == data_image[mid_2[1], mid_2[0], 2]:
                    neighbors_same_cell[i, 0] = 1

            if top_index != -1:
                token_rect_2 = all_tokens_rects[top_index]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share row
                if data_image[mid[1], mid[0], 0] == data_image[mid_2[1], mid_2[0], 0]:
                    neighbors_same_row[i, 1] = 1
                # They share column
                if data_image[mid[1], mid[0], 1] == data_image[mid_2[1], mid_2[0], 1]:
                    neighbors_same_col[i, 1] = 1
                # They share cell
                if data_image[mid[1], mid[0], 2] == data_image[mid_2[1], mid_2[0], 2]:
                    neighbors_same_cell[i, 1] = 1

            if right_index != -1:
                token_rect_2 = all_tokens_rects[right_index]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share row
                if data_image[mid[1], mid[0], 0] == data_image[mid_2[1], mid_2[0], 0]:
                    neighbors_same_row[i, 2] = 1
                # They share column
                if data_image[mid[1], mid[0], 1] == data_image[mid_2[1], mid_2[0], 1]:
                    neighbors_same_col[i, 2] = 1
                # They share cell
                if data_image[mid[1], mid[0], 2] == data_image[mid_2[1], mid_2[0], 2]:
                    neighbors_same_cell[i, 2] = 1

            if bottom_index != -1:
                token_rect_2 = all_tokens_rects[bottom_index]
                mid_2 = [int(token_rect_2['x'] + token_rect_2['width'] / 2),
                         int(token_rect_2['y'] + token_rect_2['height'] / 2)]
                # They share row
                if data_image[mid[1], mid[0], 0] == data_image[mid_2[1], mid_2[0], 0]:
                    neighbors_same_row[i, 3] = 1
                # They share column
                if data_image[mid[1], mid[0], 1] == data_image[mid_2[1], mid_2[0], 1]:
                    neighbors_same_col[i, 3] = 1
                # They share cell
                if data_image[mid[1], mid[0], 2] == data_image[mid_2[1], mid_2[0], 2]:
                    neighbors_same_cell[i, 3] = 1

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

        sorted_path_full = self.sorted_path + "-%d" % increment
        if not dont_output:
            if not os.path.exists(sorted_path_full):
                os.mkdir(sorted_path_full)


        cv2.imwrite(os.path.join(sorted_path_full, 'visual.png'), show_1)

        # To place input vectors at respective spatial coordinates
        input_tensor = np.zeros((256, 256, 308)).astype(np.float64)
        # Same zone or not, 0 for not, 1 for yes
        output_tensor = np.zeros((256, 256, 4)).astype(np.float64)
        # Whether there was a word here or not
        # output_tensor_word_mask = np.zeros((256, 256)).astype(np.float64)
        output_tensor_word_mask = np.zeros((256, 256)).astype(np.float64)

        output_tensor_zone_mask = np.ones((256, 256), dtype=np.float32)

        table_width = tx2 - tx1
        table_height = ty2 - ty1
        rgb = np.zeros((256, 256, 3))
        glove_not_found = 0.0
        for i in range(N):
            token_rect = all_tokens_rects[i]

            # Source coordinates of top left of tokens
            cx = token_rect['x'] - tx1
            cy = token_rect['y'] - ty1
            cw = token_rect['width']
            ch = token_rect['height']

            distances_vector = D[i]

            # Get the GloVe reading
            embedding = self.glove_reader.get_vector(all_tokens[i])
            if embedding is None:
                embedding = np.ones((300)) * (-1)
                glove_not_found += 1

            positional = np.array([cx / table_width, cy / table_height, cw / table_width, ch / table_height,
                                   distances_vector[0] / table_width, distances_vector[1] / table_height,
                                   distances_vector[2] / table_width, distances_vector[3] / table_height])

            # Destination coordinates on 256x256 scale and place there
            nx = math.floor(256.0 * cx / table_width)
            ny = math.floor(256.0 * cy / table_height)
            input_tensor[ny, nx] = np.concatenate((embedding, positional))

            # From the neighbor graph
            output_tensor[ny, nx] = np.array(
                [neighbors_same_cell[i, 0], neighbors_same_cell[i, 1], neighbors_same_cell[i, 2],
                 neighbors_same_cell[i, 3]])

            if neighbors_same_cell[i, 0] == 1 or neighbors_same_cell[i, 1] == 1:
                rgb[ny,nx] = np.array([0,0,255])
            else:
                rgb[ny,nx] = np.array([255,255,255])
                # Set mask to 1
                # output_tensor_word_mask[ny, nx] =1
                # print (output_tensor_word_mask[ny, nx])

            output_tensor_word_mask[ny, nx] = 1


        if glove_not_found / N > 0.3:
            print("WARNING: GloVe not found ratio", glove_not_found / N)

        # Output debugging visual file for zone mask
        segmentation_visualize_path = os.path.join(sorted_path_full, 'visual_segment.png')
        cv2.imwrite(segmentation_visualize_path, (output_tensor_zone_mask * 255).astype(np.uint8))

        # Output debugging visual image for word mask
        word_mask_path = os.path.join(sorted_path_full, 'visual_word_mask.png')
        output_tensor_word_mask_temp = (rgb.transpose((2, 0, 1)) * output_tensor_zone_mask).transpose(1, 2, 0)
        # output_tensor_word_mask_temp=rgb*np.repeat(output_tensor_zone_mask,3).reshape((256,256,3))

        # output_tensor_zone_mask_temp  = np.resize(output_tensor_zone_mask, (256, 256, 3))

        # output_tensor_word_mask=np.multiply(rgb,output_tensor_zone_mask_temp )
        cv2.imwrite(word_mask_path, rgb.astype(np.uint8))
        word_mask_path_1 = os.path.join(sorted_path_full, 'visual_word_mask_masked.png')
        cv2.imwrite(word_mask_path_1, output_tensor_word_mask_temp.astype(np.uint8))
        # cv2.imwrite(word_mask_path, (output_tensor_word_mask *255).astype(np.uint8))

        cv2.imwrite(os.path.join(sorted_path_full,'table_cropped.png'), image_table_cropped)

        # Dump the content to pickle file. The file is compressed by gzip.
        dump_path = os.path.join(sorted_path_full, '__dump__.pklz')
        document = TableParseDocument(input_tensor, output_tensor, output_tensor_word_mask, output_tensor_zone_mask)
        f = gzip.open(dump_path, 'wb')
        pickle.dump(document, f)
        f.close()

    def do_plot(self, document, id):
        rects = document.rects
        row_share = document.row_share
        canvas = (np.ones((500,500, 3))*255).astype(np.uint8)
        for i in range(len(rects)):
            rect = rects[i]
            color = (255, 0, 0) if document.row_share[0, i] == 0 else (0,0,255)
            cv2.rectangle(canvas, (int(rect[0] * 500), int(rect[1]*500)), (int((rect[0]+rect[2]) * 500), int((rect[1]+rect[3])*500)), color)
        cv2.imshow('test' + id, canvas)
        cv2.waitKey(0)

    def dump_table(self, all_tokens, all_tokens_rects, neighbor_graph, neighbor_distance_matrix, share_row_matrix,
                   share_col_matrix, share_cell_matrix, neighbors_same_row, neighbors_same_col, neighbors_same_cell, image_visual, spatial_features, file_name):
        N = len(all_tokens)
        height, width, _ = np.shape(image_visual)
        classes = np.zeros(N)
        rect_matrix = np.zeros((N, 4))
        embeddings_matrix = np.zeros((N, 300))
        features_spatial_height, features_spatial_width, depth = np.shape(spatial_features)

        conv_features =  np.zeros((N,depth))

        for i in range(N):
            token_rect = all_tokens_rects[i]
            rect_matrix[i, 0] = token_rect['x'] / width
            rect_matrix[i, 1] = token_rect['y'] / height
            rect_matrix[i, 2] = token_rect['width'] / width
            rect_matrix[i, 3] = token_rect['height'] / height

            feat_x = int((rect_matrix[i, 0] + rect_matrix[i, 2] / 2) * features_spatial_width)
            feat_y = int((rect_matrix[i, 1] + rect_matrix[i, 3] / 2) * features_spatial_height)

            assert feat_x < features_spatial_width and feat_y < features_spatial_height

            conv_features[i] = spatial_features[feat_y, feat_x]

            embedding = glove_reader.get_vector(all_tokens[i])
            if embedding is None:
                embedding = np.ones((300)) * (-1)
            embeddings_matrix[i] = embedding

        document = TableData(embeddings_matrix, rect_matrix, neighbor_distance_matrix, neighbor_graph, share_row_matrix,
                             share_col_matrix, share_cell_matrix, neighbors_same_row, neighbors_same_col,
                             neighbors_same_cell, conv_features)

        if show:
            self.do_plot(document, file_name)


        if not dont_output:
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

    UnlvConverter(id, png_path_full, xml_path_full, ocr_path_full, sorted_path_full, glove_reader).execute()
