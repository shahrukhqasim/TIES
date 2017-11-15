import numpy as np
import torch
from torch.autograd import Variable
from network.ModuleCollect import ModuleCollect
from table_detect.dense import Dense


class TableDetect(torch.nn.Module):
    def __init__(self, num_embeddings, num_positional, num_convolutional):
        super(TableDetect, self).__init__()

        self.moduleA_embeddings = Dense(num_embeddings, config = [300, 'R', 300, 'R', 100])
        self.moduleA_positional = Dense(num_positional, config = [20, 'S', 20, 'R', 100])
        self.moduleA_convolutional = Dense(num_convolutional, config = [100, 'R', 100, 'R', 100])

        self.moduleA_project_down = Dense(300, config=[100, 'R'])

        self.moduleB = torch.nn.GRUCell(500, 100).cuda()
        self.moduleBO_1 = Dense(100, config = [100, 'S', 100, 'S', 100, 'S'])
        self.moduleBO_2 = Dense(100, config = [100, 'S', 100, 'S', 100, 'S'])
        self.moduleC = Dense(100, config = [100])
        self.moduleD = Dense(100, config = [100, 'S'])

        self.iterations = 1

    def set_iterations(self, iterations):
        self.iterations = iterations

    def concat(self, x, indices, indices_not_found, num_words):
        y = Variable(torch.zeros(num_words, 100 * 5)).cuda()
        y[:, 000:100] = x[indices[:, 0]]
        y[:, 100:200] = x[indices[:, 1]]
        y[:, 200:300] = x[indices[:, 2]]
        y[:, 300:400] = x[indices[:, 3]]
        y[:, 400:500] = x[indices[:, 4]]
        y[indices_not_found] = 0

        return y

    def forward(self, indices, indices_not_found, word_embeddings, positional_features, convolutional_features, num_words):
        uu_embeddings = self.moduleA_embeddings.forward(word_embeddings)
        uu_positional = self.moduleA_positional.forward(positional_features)
        uu_convolutional = self.moduleA_convolutional.forward(convolutional_features)

        uu_combined = torch.cat((uu_embeddings, uu_positional, uu_convolutional), dim=1)
        uu = self.moduleA_project_down.forward(uu_combined)

        hh = Variable(torch.zeros(num_words,100)).cuda()

        for i in range(self.iterations):
            ww = self.concat(uu, indices, indices_not_found, num_words)
            bb = self.moduleB.forward(ww, hh)
            oo, hh = self.moduleBO_1.forward(bb), self.moduleBO_2.forward(bb)
            ll = self.moduleC.forward(oo)
            uu = self.moduleD.forward(hh)

        return ll
