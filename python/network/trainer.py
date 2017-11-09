import numpy as np
import configparser as cp
from network.data_features_dumper import DataFeaturesDumper
from network.silknet import LoadInterface
from network.silknet.FolderDataReader import FolderDataReader
from interface import implements
import cv2
import os
import pickle
from network.computation_graph import SimpleDocProcModel


class DataLoader(implements(LoadInterface)):
    def load_datum(self, full_path):
        with open(os.path.join(full_path, '__dump__.pickle'), 'rb') as f:
            doc = pickle.load(f)
        return doc


class Trainer:
    def __init__(self):
        config = cp.ConfigParser()
        config.read('config.ini')
        self.train_path = config['quad']['train_data_path']
        self.test_path = config['quad']['test_data_path']
        self.glove_path = config['quad']['glove_path']

    def init(self, dump_features_again):
        if dump_features_again:
            self.reader = DataFeaturesDumper(self.train_path, self.glove_path, 'train')
            self.reader.load()
            self.glove_reader = self.reader.get_glove_reader()

    def train(self):
        dataset = FolderDataReader(self.train_path, DataLoader())
        dataset.init()
        model = SimpleDocProcModel()
        for i in range(300):
            document, epoch, id = dataset.next_element()
            num_words, _ = np.shape(document.tokens_rects)
            for j in range(300):
                vv = np.concatenate([document.rects, document.distances, document.embeddings], axis=1)
                model.forward(document.tokens_neighbor_matrix, vv, num_words)