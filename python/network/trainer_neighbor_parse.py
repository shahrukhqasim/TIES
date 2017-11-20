import numpy as np
import configparser as cp
from network.data_features_dumper import DataFeaturesDumper
from network.silknet import LoadInterface
from network.silknet.FolderDataReader import FolderDataReader
from interface import implements
import os
import pickle
from network.computation_graph_neighbor_parse import ComputationGraphTableNeighborParse
import torch
from torch.autograd import Variable
import cv2


class DataLoader(implements(LoadInterface)):
    def load_datum(self, full_path):
        with open(os.path.join(full_path, '__dump__.pickle'), 'rb') as f:
            doc = pickle.load(f)
        return doc


class Trainer:
    def __init__(self):
        config = cp.ConfigParser()
        config.read('config.ini')
        self.train_path = config['neighbor_parse']['train_data_path']
        self.test_path = config['neighbor_parse']['test_data_path']
        self.validation_data_path = config['neighbor_parse']['validation_data_path']
        self.glove_path = config['neighbor_parse']['glove_path']
        self.learning_rate = float(config['neighbor_parse']['learning_rate'])
        self.from_scratch = int(config['neighbor_parse']['from_scratch']) == 1
        self.model_path = config['neighbor_parse']['model_path']
        self.save_after = int(config['neighbor_parse']['save_after'])

    def init(self, dump_features_again):
        pass

    def do_plot(self, document, id):
        rects = document.rects
        row_share = document.row_share
        canvas = (np.ones((500,500, 3))*255).astype(np.uint8)
        for i in range(len(rects)):
            rect = rects[i]
            color = (255, 0, 0) if document.cell_share[0, i] == 0 else (0,0,255)
            cv2.rectangle(canvas, (int(rect[0] * 500), int(rect[1]*500)), (int((rect[0]+rect[2]) * 500), int((rect[1]+rect[3])*500)), color)
        cv2.imshow('test' + id, canvas)
        cv2.waitKey(0)

    def get_example_elements(self, document, id):
        num_words, _ = np.shape(document.rects)
        vv = np.concatenate([document.rects, document.distances, document.embeddings], axis=1).astype(np.float32)
        vv = Variable(torch.from_numpy(vv)).cuda()
        y = Variable(torch.from_numpy(document.neighbors_same_cell[:,0].astype(np.int64)), requires_grad=False).cuda()

        baseline_accuracy_1 = 100 * np.sum(document.neighbors_same_cell[:,0] == 0) / num_words
        baseline_accuracy_2 = 100 * np.sum(document.neighbors_same_cell[:,0] == 1) / num_words

        indices = torch.LongTensor(torch.from_numpy(np.concatenate(
            [np.expand_dims(np.arange(num_words, dtype=np.int64), axis=1),
             np.maximum(document.neighbor_graph.astype(np.int64), 0)], axis=1))).cuda()
        indices_not_found = torch.ByteTensor(torch.from_numpy(np.repeat(np.concatenate(
            [np.expand_dims(np.zeros(num_words, dtype=np.int64), axis=1),
             document.neighbor_graph.astype(np.int64)], axis=1) == -1, 100).reshape((-1, 500)).astype(
            np.uint8))).cuda()
        # indices_not_found = indices_not_found * 0

        return num_words, vv, y, baseline_accuracy_1, baseline_accuracy_2, indices, indices_not_found

    def do_validation(self, model, dataset):

        sum_of_accuracies = 0
        total = 0
        while True:
            document, epoch, id = dataset.next_element()
            num_words, vv, y, baseline_accuracy_1, baseline_accuracy_2, indices, indices_not_found = self.get_example_elements(document, id)

            y_pred = model(indices, indices_not_found, vv, num_words)
            _, predicted = torch.max(y_pred.data, 1)

            accuracy = torch.sum(predicted == y.data)
            accuracy = 100 * accuracy / num_words

            print(accuracy)

            total += 1
            sum_of_accuracies += accuracy

            if epoch == 1:
                break

        print("Average validation accuracy = ", sum_of_accuracies / total)

    def train(self):

        dataset = FolderDataReader(self.train_path, DataLoader())
        validation_dataset = FolderDataReader(self.validation_data_path, DataLoader())
        dataset.init()
        validation_dataset.init()
        model = ComputationGraphTableNeighborParse()
        model.set_iterations(4)
        criterion = torch.nn.CrossEntropyLoss(size_average=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for i in range(1000000):
            if i % 10000 == 0:
                self.do_validation(model, validation_dataset)

            document, epoch, id = dataset.next_element()
            num_words, vv, y, baseline_accuracy_1, baseline_accuracy_2, indices, indices_not_found = self.get_example_elements(document, id)


            for j in range(1):
                y_pred = model(indices, indices_not_found, vv, num_words)
                _, predicted = torch.max(y_pred.data, 1)
                accuracy = torch.sum(predicted == y.data)
                accuracy = 100 * accuracy / num_words

                yes_pred = torch.sum(predicted == 0)
                yes_pred = 100 * yes_pred / num_words

                no_pred = torch.sum(predicted == 1)
                no_pred = 100 * no_pred / num_words


                loss = criterion(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("%3dx%3d Loss = %f" %  (i, j, loss.data[0]), "Accuracy: %03.2f" % accuracy, "Yes: %03.2f" % yes_pred, "No: %03.2f" % no_pred, "Base Yes: %03.2f" % baseline_accuracy_1,"Base No: %03.2f" % baseline_accuracy_2, torch.sum(y_pred).data[0])
