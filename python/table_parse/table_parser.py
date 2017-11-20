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


class DataLoader(implements(LoadInterface)):
    def load_datum(self, full_path):
        with open(os.path.join(full_path, '__dump__.pickle'), 'rb') as f:
            doc = pickle.load(f)
        return doc


class TableParser:
    def __init__(self):
        config = cp.ConfigParser()
        config.read('config.ini')
        self.train_path = config['table_parse']['train_data_path']
        self.test_path = config['table_parse']['test_data_path']
        self.validation_data_path = config['table_parse']['validation_data_path']
        self.learning_rate = float(config['table_parse']['learning_rate'])
        self.save_after = int(config['table_parse']['save_after'])
        self.model_path = config['table_parse']['model_path']
        self.from_scratch = int(config['table_parse']['from_scratch']) == 1
        self.raw_images_path = config['table_parse']['raw_images_path']

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

        all_y = []
        all_baseline_1 = []
        all_baseline_2 = []

        for i in range(4):
            y = Variable(torch.from_numpy(document.neighbors_same_cell[:,i].astype(np.int64)), requires_grad=False).cuda()
            baseline_accuracy_1 = 100 * np.sum(document.neighbors_same_cell[:,i] == 0) / num_words
            baseline_accuracy_2 = 100 * np.sum(document.neighbors_same_cell[:,i] == 1) / num_words
            all_y.append(y)
            all_baseline_1.append(baseline_accuracy_1)
            all_baseline_2.append(baseline_accuracy_2)

        indices = torch.LongTensor(torch.from_numpy(np.concatenate(
            [np.expand_dims(np.arange(num_words, dtype=np.int64), axis=1),
             np.maximum(document.neighbor_graph.astype(np.int64), 0)], axis=1))).cuda()
        indices_not_found = torch.ByteTensor(torch.from_numpy(np.repeat(np.concatenate(
            [np.expand_dims(np.zeros(num_words, dtype=np.int64), axis=1),
             document.neighbor_graph.astype(np.int64)], axis=1) == -1, 100).reshape((-1, 500)).astype(
            np.uint8))).cuda()
        # indices_not_found = indices_not_found * 0

        return num_words, vv_positional, vv_embeddings, vv_convolutional, all_y, all_baseline_1, all_baseline_2, indices, indices_not_found

    def do_validation(self, model_cell_top, model_cell_left, dataset):

        sum_of_accuracies_top = 0
        sum_of_accuracies_left = 0
        last_epoch = dataset.get_next_epoch()
        total = 0
        while True:
            document, epoch, id = dataset.next_element()
            num_words, vv_positional, vv_embeddings, vv_convolutional, all_y, all_baseline_1, all_baseline_2, indices, indices_not_found = self.get_example_elements(document, id)


            y_pred_top = model_cell_top(indices, indices_not_found, vv_embeddings, vv_positional, vv_convolutional,
                                        num_words)
            y_pred_left = model_cell_left(indices, indices_not_found, vv_embeddings, vv_positional, vv_convolutional,
                                         num_words)
            _, predicted_top = torch.max(y_pred_top.data, 1)
            _, predicted_left = torch.max(y_pred_left.data, 1)

            accuracy_top = torch.sum(predicted_top == all_y[1].data)
            accuracy_top = 100 * accuracy_top / num_words

            accuracy_left = torch.sum(predicted_left == all_y[0].data)
            accuracy_left = 100 * accuracy_left / num_words

            self.plot_result_cell(id, document.rects, document.neighbor_graph, document.neighbors_same_cell, y_pred_left.cpu().data.numpy(), y_pred_top.cpu().data.numpy())

            print("Top", accuracy_top, "Left", accuracy_left)

            total += 1
            sum_of_accuracies_top += accuracy_top
            sum_of_accuracies_left += accuracy_left

            if epoch == last_epoch+1:
                break

        print("Average validation accuracy = ", sum_of_accuracies_top / total)
        print("Average validation accuracy = ", sum_of_accuracies_left / total)
        input()

    def plot_result_cell(self, id, rects_matrix, neighbor_matrix, share_cell_matrix, share_left, share_top):
        raw_image_full_path = os.path.join(self.raw_images_path, os.path.splitext(id)[0] + '.png')
        print(raw_image_full_path)
        image = cv2.imread(raw_image_full_path, 1)
        N, _ = np.shape(rects_matrix)
        height, width, _ = np.shape(image)

        share_top = np.argmax(share_top, axis=1)
        share_left = np.argmax(share_left, axis=1)

        neighbor_matrix = np.copy(neighbor_matrix).astype(np.int32)

        for i in range(N):
            union_rect = rects_matrix[i]
            # if share_cell_matrix[i, 0] == 1:
            #     union_rect = helpers.rects_functions.union(rects_matrix[neighbor_matrix[i, 0]], union_rect)
            # if share_cell_matrix[i, 1] == 1:
            #     union_rect = helpers.rects_functions.union(rects_matrix[neighbor_matrix[i, 1]], union_rect)
            # if share_cell_matrix[i, 2] == 1:
            #     union_rect = helpers.rects_functions.union(rects_matrix[neighbor_matrix[i, 2]], union_rect)
            # if share_cell_matrix[i, 3] == 1:
            #     union_rect = helpers.rects_functions.union(rects_matrix[neighbor_matrix[i, 3]], union_rect)

            if share_top[i] == 1:
                union_rect = helpers.rects_functions.union(rects_matrix[neighbor_matrix[i, 1]], union_rect)
            if share_left[i] == 1:
                union_rect = helpers.rects_functions.union(rects_matrix[neighbor_matrix[i, 0]], union_rect)

            left = int(union_rect[0] * width)
            top = int(union_rect[1] * height)
            right = int(union_rect[2] * width + left)
            bottom = int(union_rect[3] * height + top)

            cv2.rectangle(image, (left, top), (right, bottom), (255,0,0), 3)

        image = cv2.resize(image, None, fx=0.35, fy=0.35)
        print(id)
        cv2.imshow('a', image)
        cv2.waitKey(0)

    # @param orientation:
    # 1. 0 Left
    # 2. 1 Top
    # 3. 2 Right
    # 4. 3 Bottom
    def get_name(self, name, orientation):
        if orientation == 0:
            return self.model_path + '_'+ name + '_left.pth'
        elif orientation == 1:
            return self.model_path + '_'+ name + '_top.pth'
        elif orientation == 2:
            return self.model_path + '_'+ name + '_right.pth'
        elif orientation == 3:
            return self.model_path + '_'+ name + '_bottom.pth'
        else:
            0/0


    def train(self):
        dataset = FolderDataReader(self.train_path, DataLoader())
        validation_dataset = FolderDataReader(self.validation_data_path, DataLoader())
        dataset.init()
        validation_dataset.init()
        model_cell_top = TableParse(300, 8, 48).cuda()
        model_cell_top.set_iterations(2)
        model_cell_left = TableParse(300, 8, 48).cuda()
        model_cell_left.set_iterations(2)
        criterion = torch.nn.CrossEntropyLoss(size_average=True)
        optimizer_top = torch.optim.Adam(model_cell_top.parameters(), lr=self.learning_rate)
        optimizer_left = torch.optim.Adam(model_cell_left.parameters(), lr=self.learning_rate)

        if not self.from_scratch:
            model_cell_left.load_state_dict(torch.load(self.get_name('cell', 0)))
            model_cell_top.load_state_dict(torch.load(self.get_name('cell', 1)))


        for i in range(1000000):
            if i % 1000 == 0:
                self.do_validation(model_cell_top, model_cell_left, validation_dataset)


            document, epoch, id = dataset.next_element()
            num_words, vv_positional, vv_embeddings, vv_convolutional, all_y, all_baseline_1, all_baseline_2, indices, indices_not_found = self.get_example_elements(document, id)

            # if i % self.save_after == 0:
            #     print("Saving model")
            #     torch.save(model_cell_left.state_dict(), self.get_name('cell', 0))
            #     torch.save(model_cell_top.state_dict(), self.get_name('cell', 1))

            for j in range(1):
                y_pred_left = model_cell_left(indices, indices_not_found, vv_embeddings, vv_positional, vv_convolutional, num_words)
                y_pred_top = model_cell_top(indices, indices_not_found, vv_embeddings, vv_positional, vv_convolutional, num_words)
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

                loss_top = criterion(y_pred_top, all_y[1])
                optimizer_top.zero_grad()
                loss_top.backward()
                optimizer_top.step()

                loss_left = criterion(y_pred_left, all_y[0])
                optimizer_left.zero_grad()
                loss_left.backward()
                optimizer_left.step()

                print("LEFT %d %s - L: %03.4f, A: %03.2f, Y: %03.2f N: %03.2f B1: %03.2f B2: %03.2f" % (i, id, loss_left.data[0], accuracy_left, yes_pred_left, no_pred_left, all_baseline_1[0] , all_baseline_2[0]))
                print("TOP  %d %s - L: %03.4f, A: %03.2f, Y: %03.2f N: %03.2f B1: %03.2f B2: %03.2f" % (i, id, loss_top.data[0], accuracy_top, yes_pred_top, no_pred_top, all_baseline_1[1] , all_baseline_2[1]))

if __name__ == '__main__':
    trainer = TableParser()
    trainer.train()