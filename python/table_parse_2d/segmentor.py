import numpy as np
import configparser as cp
from network.silknet import LoadInterface
from network.silknet.FolderDataReader import FolderDataReader
from interface import implements
import os
import pickle
from table_parse.computation_graph_table_parse import TableParse
import torch
from torch.autograd import Variable
import cv2
import helpers.rects_functions
import random
import gzip

class DataLoader(implements(LoadInterface)):
    def load_datum(self, full_path):
        # The file is compressed, so load it using gzip
        f = gzip.open(os.path.join(full_path, '__dump__.pickle'), 'rb')
        doc = pickle.load(f)
        f.close()

        return doc


class Segmentor:
    def __init__(self):
        config = cp.ConfigParser()
        config.read('config.ini')
        self.train_path = config['zone_segment']['train_data_path']
        self.test_path = config['zone_segment']['test_data_path']
        self.validation_data_path = config['zone_segment']['validation_data_path']
        self.learning_rate = float(config['zone_segment']['learning_rate'])
        self.save_after = int(config['zone_segment']['save_after'])
        self.model_path = config['zone_segment']['model_path']
        self.from_scratch = int(config['zone_segment']['from_scratch']) == 1

    def train(self):
        dataset = FolderDataReader(self.train_path, DataLoader())
        dataset.init()
        validation_dataset = FolderDataReader(self.validation_data_path, DataLoader())
        validation_dataset.init()
        model = TableParse(300, 8, 48).cuda()
        criterion = torch.nn.CrossEntropyLoss(size_average=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        if not self.from_scratch:
            model.load_state_dict(self.model_path)


        for i in range(1000000):
            if i % 1000 == 0:
                self.do_validation(model_cell_top, model, validation_dataset, True)


            document, epoch, id = dataset.next_element()
            num_words, vv_positional, vv_embeddings, vv_convolutional, all_y, all_baseline_1, all_baseline_2, indices, indices_not_found = self.get_example_elements(document, id, augment=True)

            if i % self.save_after == 0:
                print("Saving model")
                torch.save(model.state_dict(), self.get_name('cell', 0))
                torch.save(model_cell_top.state_dict(), self.get_name('cell', 1))

            for j in range(1):
                y_pred_lefts = model(indices, indices_not_found, vv_embeddings, vv_positional, vv_convolutional, num_words)
                y_pred_tops = model_cell_top(indices, indices_not_found, vv_embeddings, vv_positional, vv_convolutional, num_words)

                y_pred_left = y_pred_lefts[iters_left - 1]
                y_pred_top = y_pred_tops[iters_top - 1]
                _, predicted_left = torch.max(y_pred_left.data, 1)
                _, predicted_top = torch.max(y_pred_top.data, 1)

                accuracy_top = torch.sum(predicted_top == all_y[1].data)
                accuracy_top = 100 * accuracy_top / num_words

                accuracy_left = torch.sum(predicted_left == all_y[0].data)
                accuracy_left = 100 * accuracy_left / num_words

                yes_pred_top = torch.sum(predicted_top == 0)
                yes_pred_top = 100 * yes_pred_top / num_words

                yes_pred_left = torch.sum(predicted_left == 0)
                yes_pred_left = 100 * yes_pred_left / num_words

                no_pred_top = torch.sum(predicted_top == 1)
                no_pred_top = 100 * no_pred_top / num_words

                no_pred_left = torch.sum(predicted_left == 1)
                no_pred_left = 100 * no_pred_left / num_words

                # print(y_pred_tops[0].size())
                loss_top = criterion(y_pred_tops[0], all_y[1])
                for k in range(iters_top-1):
                    loss_top += criterion(y_pred_tops[k+1], all_y[1]) * (k+1) * (k+1)
                optimizer_top.zero_grad()
                loss_top.backward()
                optimizer_top.step()

                loss_left = criterion(y_pred_lefts[0], all_y[0])
                for k in range(iters_left-1):
                    loss_left += criterion(y_pred_lefts[k+1], all_y[1]) * (k+1) * (k+1)
                optimizer.zero_grad()
                loss_left.backward()
                optimizer.step()

                print("LEFT %d %s - L: %03.4f, A: %03.2f, Y: %03.2f N: %03.2f B1: %03.2f B2: %03.2f" % (i, id, loss_left.data[0], accuracy_left, yes_pred_left, no_pred_left, all_baseline_1[0] , all_baseline_2[0]))
                print("TOP  %d %s - L: %03.4f, A: %03.2f, Y: %03.2f N: %03.2f B1: %03.2f B2: %03.2f" % (i, id, loss_top.data[0], accuracy_top, yes_pred_top, no_pred_top, all_baseline_1[1] , all_baseline_2[1]))

if __name__ == '__main__':
    segmentor = Segmentor()
    segmentor.train()