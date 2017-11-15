import numpy as np
import configparser as cp
from network.silknet import LoadInterface
from network.silknet.FolderDataReader import FolderDataReader
from interface import implements
import os
import pickle
from table_detect.computation_graph_table_detect import TableDetect
import torch
from torch.autograd import Variable
import cv2


class DataLoader(implements(LoadInterface)):
    def load_datum(self, full_path):
        with open(os.path.join(full_path, '__dump__.pickle'), 'rb') as f:
            doc = pickle.load(f)
        return doc


class TableDetector:
    def __init__(self):
        config = cp.ConfigParser()
        config.read('config.ini')
        self.train_path = config['table_detect']['train_data_path']
        self.test_path = config['table_detect']['test_data_path']
        self.validation_data_path = config['table_detect']['validation_data_path']
        self.learning_rate = float(config['table_detect']['learning_rate'])

    def do_plot(self, document, id):
        rects = document.rects
        classes = document.classes
        canvas = (np.ones((500,500, 3))*255).astype(np.uint8)
        for i in range(len(rects)):
            rect = rects[i]
            color = (255, 0, 0) if classes[i] == 0 else (0,0,255)
            cv2.rectangle(canvas, (int(rect[0] * 500), int(rect[1]*500)), (int((rect[0]+rect[2]) * 500), int((rect[1]+rect[3])*500)), color)
        cv2.imshow('test' + id, canvas)
        cv2.waitKey(0)

    def get_example_elements(self, document, id):
        num_words, _ = np.shape(document.rects)
        vv_positional = np.concatenate([document.rects, document.distances], axis=1).astype(np.float32)
        vv_positional = Variable(torch.from_numpy(vv_positional)).cuda()

        vv_embeddings = document.embeddings.astype(np.float32)
        vv_embeddings = Variable(torch.from_numpy(vv_embeddings)).cuda()

        vv_convolutional = document.conv_features.astype(np.float32)
        vv_convolutional = Variable(torch.from_numpy(vv_convolutional)).cuda()


        y = Variable(torch.from_numpy(document.classes.astype(np.int64)), requires_grad=False).cuda()

        baseline_accuracy_1 = 100 * np.sum(document.classes == 0) / num_words
        baseline_accuracy_2 = 100 * np.sum(document.classes == 1) / num_words

        indices = torch.LongTensor(torch.from_numpy(np.concatenate(
            [np.expand_dims(np.arange(num_words, dtype=np.int64), axis=1),
             np.maximum(document.neighbor_graph.astype(np.int64), 0)], axis=1))).cuda()
        indices_not_found = torch.ByteTensor(torch.from_numpy(np.repeat(np.concatenate(
            [np.expand_dims(np.zeros(num_words, dtype=np.int64), axis=1),
             document.neighbor_graph.astype(np.int64)], axis=1) == -1, 100).reshape((-1, 500)).astype(
            np.uint8))).cuda()
        # indices_not_found = indices_not_found * 0

        return num_words, vv_positional, vv_embeddings, vv_convolutional, y, baseline_accuracy_1, baseline_accuracy_2, indices, indices_not_found

    def do_validation(self, model, dataset):

        sum_of_accuracies = 0
        total = 0
        while True:
            document, epoch, id = dataset.next_element()
            num_words, vv_positional, vv_embeddings, vv_convolutional, y, baseline_accuracy_1, baseline_accuracy_2, indices, indices_not_found = self.get_example_elements(document, id)

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
        # validation_dataset = FolderDataReader(self.validation_data_path, DataLoader())
        dataset.init()
        # validation_dataset.init()
        model = TableDetect(300,8,48).cuda()
        model.set_iterations(2)
        criterion = torch.nn.CrossEntropyLoss(size_average=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for i in range(1000000):
            # if i % 10000 == 0:
            #     self.do_validation(model, validation_dataset)

            document, epoch, id = dataset.next_element()
            num_words, vv_positional, vv_embeddings, vv_convolutional, y, baseline_accuracy_1, baseline_accuracy_2, indices, indices_not_found = self.get_example_elements(document, id)


            for j in range(2):
                y_pred = model(indices, indices_not_found, vv_embeddings, vv_positional, vv_convolutional, num_words)
                _, predicted = torch.max(y_pred.data, 1)
                accuracy = torch.sum(predicted == y.data)
                accuracy = 100 * accuracy / num_words

                tables_pred = torch.sum(predicted == 0)
                tables_pred = 100 * tables_pred / num_words

                non_tables_pred = torch.sum(predicted == 1)
                non_tables_pred = 100 * non_tables_pred / num_words


                loss = criterion(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("%3dx%3d Loss = %f" %  (i, j, loss.data[0]), "Accuracy: %03.2f" % accuracy, "Tables: %03.2f" % tables_pred, "Non-tables: %03.2f" % non_tables_pred, "Base 1: %03.2f" % baseline_accuracy_1,"Base 2: %03.2f" % baseline_accuracy_2, torch.sum(y_pred).data[0])

if __name__ == '__main__':
    trainer = TableDetector()
    trainer.train()