import os
import nltk
import ssl
import numpy as np
import random
from .glove_reader import GLoVe
import cv2
import json
from network.neighbor_graph_builder import NeighborGraphBuilder
from random import shuffle
from network.document_features import DocumentFeatures
import pickle


# TODO: Tackle - Won't work for very large dataset because images are loaded into memory
class DataFeaturesDumper:
    path = ''
    tokens_set = set()
    docs = []
    queue = []

    def __init__(self, path, glove_path, cache_name):
        self.path = path
        self.glove_path = glove_path
        self.cache_name = cache_name

    def dump_doc(self, all_tokens, all_tokens_rects, image, file_name):
        N = len(all_tokens)
        height, width = np.shape(image)
        classes = np.zeros(N)
        rect_matrix = np.zeros((N, 4))
        embeddings_matrix = np.zeros((N, 300))
        for i in range(N):
            token_rect = all_tokens_rects[i]
            index = 0 if image[int(token_rect['y'] + token_rect['height'] / 2), int(
                token_rect['x'] + token_rect['width'] / 2)] == 0 else 1
            classes[i] = index
            rect_matrix[i, 0] = token_rect['x'] / width
            rect_matrix[i, 1] = token_rect['y'] / height
            rect_matrix[i, 2] = token_rect['width'] / width
            rect_matrix[i, 3] = token_rect['height'] / height
            embedding = self.glove_reader.get_vector(all_tokens[i])
            if embedding is None:
                embedding = np.ones((300)) * (-1)
            embeddings_matrix[i] = embedding


        graph_builder = NeighborGraphBuilder(all_tokens_rects, image)
        neighbor_graph, neighbor_distance_matrix = graph_builder.get_neighbor_matrix()
        neighbor_distance_matrix[:, 0] = neighbor_distance_matrix[:, 0] / width
        neighbor_distance_matrix[:, 1] = neighbor_distance_matrix[:, 1] / height
        neighbor_distance_matrix[:, 2] = neighbor_distance_matrix[:, 2] / width
        neighbor_distance_matrix[:, 3] = neighbor_distance_matrix[:, 3] / height
        document = DocumentFeatures(embeddings_matrix, rect_matrix, neighbor_distance_matrix, neighbor_graph, classes)
        with open(file_name, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

    def get_glove_reader(self):
        return self.glove_reader

    def build_glove(self):
        # Find all the unique words first
        ii = 0
        for i in os.listdir(self.path):
            full_example_path = os.path.join(self.path, i)

            json_path  = os.path.join(full_example_path, 'ocr_gt.json')

            with open(json_path) as data_file:
                ocr_data = json.load(data_file)

            for i in range(len(ocr_data)):
                word_data = ocr_data[i]
                tokens = nltk.word_tokenize(word_data['word'])
                for j in tokens:
                    self.tokens_set.add(j)
        print("Unique words are %d" % len(self.tokens_set))
        self.glove_reader = GLoVe(self.glove_path, self.tokens_set)
        self.glove_reader.load(self.cache_name)

    def load(self):
        print("Loading data")
        self.build_glove()

        ii = 0
        for i in os.listdir(self.path):
            print("On", i)
            full_example_path = os.path.join(self.path, i)

            image_path = os.path.join(full_example_path, 'tables.png')
            json_path  = os.path.join(full_example_path, 'ocr_gt.json')
            document_dump_path  = os.path.join(full_example_path, '__dump__.pickle')
            image = cv2.imread(image_path, 0)
            height, width = np.shape(image)

            all_words = []
            all_words_rects = []

            with open(json_path) as data_file:
                ocr_data = json.load(data_file)

            for i in range(len(ocr_data)):
                word_data = ocr_data[i]
                all_words.append(word_data['word'])
                all_words_rects.append(word_data['rect'])

            all_tokens = []
            all_tokens_rects = []
            class_indices = []

            for i in range(len(all_words)):
                tokens = nltk.word_tokenize(all_words[i])
                all_tokens.extend(tokens)
                word_rect = all_words_rects[i]
                divided_width = word_rect['width'] / len(tokens)
                # If a word contains more than one token, just
                # divide along width
                for j in range(len(tokens)):
                    token_rect = dict(word_rect)
                    token_rect['x'] += int(j*divided_width)
                    token_rect['width'] = int(divided_width)
                    all_tokens_rects.append(token_rect)
            assert(len(all_tokens) == len(all_tokens_rects))

            for i in range(len(all_tokens)):
                token_rect = all_tokens_rects[i]
                try:
                    class_indices.append(0 if image[int(token_rect['y'] + token_rect['height']/2), int(token_rect['x'] + token_rect['width']/2)] == 0 else 1)
                except:
                    print(i, all_tokens[i], all_tokens_rects[i])
                    pass

            self.dump_doc(all_tokens, all_tokens_rects, image, document_dump_path)
            ii += 1
            print("Loaded %d" % ii)

        print("Data loaded")