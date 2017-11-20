import numpy as np
import configparser as cp
from network.data_features_dumper import DataFeaturesDumper
from network.silknet import LoadInterface
from network.silknet.FolderDataReader import FolderDataReader
from interface import implements
import os
import pickle
from network.computation_graph_parse import ComputationGraphTableParse
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
        self.train_path = config['parse']['train_data_path']
        self.test_path = config['parse']['test_data_path']
        self.validation_data_path = config['parse']['validation_data_path']
        self.glove_path = config['parse']['glove_path']
        self.learning_rate = float(config['parse']['learning_rate'])
        self.from_scratch = int(config['parse']['from_scratch']) == 1
        self.model_path = config['parse']['model_path']
        self.save_after = int(config['parse']['save_after'])

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

    def sample_indices(self, vector):
        A = vector.copy()
        num_ones = np.sum(A == 1)

        sample_from = np.where(A == 0)[0]
        np.random.shuffle(sample_from)
        picked_zeros = sample_from[0:min(num_ones, np.size(sample_from))]
        A[picked_zeros] = 1

        return np.where(A == 1)[0]

    def get_example_elements(self, document, id):
        num_words, _ = np.shape(document.rects)
        vv = np.concatenate([document.rects, document.distances, document.embeddings], axis=1).astype(np.float32)
        vv = Variable(torch.from_numpy(vv)).cuda()

        row_share_flatten = document.row_share.flatten()
        col_share_flatten = document.col_share.flatten()
        cell_share_flatten = document.cell_share.flatten()

        row_check_indices = self.sample_indices(row_share_flatten)
        col_check_indices = self.sample_indices(col_share_flatten)
        cell_check_indices = self.sample_indices(cell_share_flatten)

        y_row = row_share_flatten[row_check_indices]
        y_col = row_share_flatten[col_check_indices]
        y_cell = row_share_flatten[cell_check_indices]

        # y_row = Variable(torch.from_numpy(document.row_share.flatten()), requires_grad=False).cuda()
        # y_col = Variable(torch.from_numpy(document.col_share.flatten().astype(np.int64)), requires_grad=False).cuda()
        # y_cell = Variable(torch.from_numpy(document.cell_share.flatten().astype(np.int64)), requires_grad=False).cuda()


        baseline_accuracy_row = 100 * np.sum(y_row == 0) / (num_words * num_words)
        baseline_accuracy_col = 100 * np.sum(y_col == 1) / (num_words * num_words)
        baseline_accuracy_cell = 100 * np.sum(y_cell == 1) / (num_words * num_words)

        y_row = Variable(torch.from_numpy(y_row.astype(np.int64))).cuda()
        y_col = Variable(torch.from_numpy(y_col.astype(np.int64))).cuda()
        y_cell = Variable(torch.from_numpy(y_cell.astype(np.int64))).cuda()

        row_check_indices = torch.from_numpy(row_check_indices).cuda()
        col_check_indices = torch.from_numpy(col_check_indices).cuda()
        cell_check_indices = torch.from_numpy(cell_check_indices).cuda()

        indices = torch.LongTensor(torch.from_numpy(np.concatenate(
            [np.expand_dims(np.arange(num_words, dtype=np.int64), axis=1),
             np.maximum(document.neighbor_graph.astype(np.int64), 0)], axis=1))).cuda()
        indices_not_found = torch.ByteTensor(torch.from_numpy(np.repeat(np.concatenate(
            [np.expand_dims(np.zeros(num_words, dtype=np.int64), axis=1),
             document.neighbor_graph.astype(np.int64)], axis=1) == -1, 100).reshape((-1, 500)).astype(
            np.uint8))).cuda()
        indices_not_found = indices_not_found * 0

        return num_words, vv, 0, 0, indices, indices_not_found, y_row, y_col, y_cell, row_check_indices, col_check_indices, cell_check_indices, baseline_accuracy_row, baseline_accuracy_col, baseline_accuracy_cell

    def do_validation(self, model, dataset):

        sum_of_accuracies = 0
        total = 0
        while True:
            document, epoch, id = dataset.next_element()
            num_words, vv, y, baseline_accuracy_1, baseline_accuracy_2, indices, indices_not_found = self.get_example_elements(document)

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
        model = ComputationGraphTableParse()

        if not self.from_scratch:
            model.load_state_dict(torch.load(self.model_path))

        # class_weights = torch.from_numpy(np.array([0.01,0.99]).astype(np.float32)).cuda()

        model.set_iterations(2)
        criterion = torch.nn.CrossEntropyLoss(size_average=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        iterations = 0

        for i in range(1000000):
            # if i % 10000 == 0:
            #     self.do_validation(model, validation_dataset)

            document, epoch, id = dataset.next_element()
            num_words, vv, baseline_accuracy_1, baseline_accuracy_2, indices, indices_not_found, y_row, y_col, y_cell, row_check_indices, col_check_indices, cell_check_indices, baseline_accuracy_row, baseline_accuracy_col, baseline_accuracy_cell = self.get_example_elements(
                document, id)

            # Save model
            if (iterations % self.save_after) == 0:
                print("Saving model")
                torch.save(model.state_dict(), self.model_path)

            for j in range(1):
                iterations += 1

                row_logits, col_logits, cell_logits = model(indices, indices_not_found, vv, num_words)
                row_logits = row_logits[row_check_indices]
                col_logits = col_logits[col_check_indices]
                cell_logits = cell_logits[cell_check_indices]

                _, predicted = torch.max(row_logits.data, 1)
                accuracy_row = torch.sum(predicted == y_row.data)
                accuracy_row = 100 * accuracy_row / (num_words * num_words)

                _, predicted = torch.max(col_logits.data, 1)
                accuracy_col = torch.sum(predicted == y_col.data)
                accuracy_col = 100 * accuracy_col / (num_words * num_words)

                _, predicted = torch.max(cell_logits.data, 1)
                accuracy_cell = torch.sum(predicted == y_cell.data)
                accuracy_cell = 100 * accuracy_cell / (num_words * num_words)

                loss_row = criterion(row_logits, y_row)
                # loss_col = criterion(col_logits, y_col)
                # loss_cell = criterion(cell_logits, y_cell)
                loss = loss_row #+ loss_col + loss_cell

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("%3dx%3d Loss = %f" % (i, j, loss.data[0]), "Row: %03.2f" % accuracy_row,
                      "BRow: %03.2f" % baseline_accuracy_row, "Col: %03.2f" % accuracy_col, "BCol: %03.2f" % baseline_accuracy_col,
                      "Cell: %03.2f" % accuracy_cell, "BCell: %03.2f" % baseline_accuracy_cell)
