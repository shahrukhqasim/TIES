import torch
import numpy as np
from torch.autograd import Variable


class ModuleCollect(torch.nn.Module):
    def __init__(self, D_in, num_words):
        super(ModuleCollect, self).__init__()
        self.D_in = D_in
        self.num_words = num_words

    def forward(self, x, indices):
        y = Variable(torch.zeros(self.num_words, self.D_in * 5))

        for i in range(self.num_words):
            y[i, 0:self.D_in] = x[i]
            y[i, self.D_in * 1:self.D_in * 2] = x[np.maximum(indices[i, 0], 0)] * int(indices[i, 0] != -1)
            y[i, self.D_in * 2:self.D_in * 3] = x[np.maximum(indices[i, 1], 0)] * int(indices[i, 1] != -1)
            y[i, self.D_in * 3:self.D_in * 4] = x[np.maximum(indices[i, 2], 0)] * int(indices[i, 2] != -1)
            y[i, self.D_in * 4:self.D_in * 5] = x[np.maximum(indices[i, 3], 0)] * int(indices[i, 3] != -1)

        return y