import numpy as np
import torch
from torch.autograd import Variable
from network.ModuleA import ModuleA
from network.ModuleB import ModuleB
from network.ModuleB2 import ModuleB2
from network.ModuleC import ModuleC
from network.ModuleD import ModuleD
from network.ModuleCollect import ModuleCollect


class SimpleDocProcModel(torch.nn.Module):
    def __init__(self):
        super(SimpleDocProcModel, self).__init__()
        self.k = 10
        self.D_in = 300 + self.k

        self.A = ModuleA(self.D_in, 100)
        self.B = ModuleB()
        self.B2 = ModuleB2(100, 100, 100)
        self.C = ModuleC(100, 2)
        self.D = ModuleD(100, 100)
        # self.Cat = ModuleCollect(100, self.N)
        self.iterations = 1

    def set_iterations(self, iterations):
        self.iterations = iterations

    def concat(self, x, indices, num_words):
        y = Variable(torch.zeros(num_words, self.D_in * 5))

        for i in range(num_words):
            y[i, 0:self.D_in] = x[i]
            y[i, self.D_in * 1:self.D_in * 2] = x[np.maximum(indices[i, 0], 0)] * int(indices[i, 0] != -1)
            y[i, self.D_in * 2:self.D_in * 3] = x[np.maximum(indices[i, 1], 0)] * int(indices[i, 1] != -1)
            y[i, self.D_in * 3:self.D_in * 4] = x[np.maximum(indices[i, 2], 0)] * int(indices[i, 2] != -1)
            y[i, self.D_in * 4:self.D_in * 5] = x[np.maximum(indices[i, 3], 0)] * int(indices[i, 3] != -1)

        return y

    def forward(self, indices, vv, num_words):
        uu = self.A.forward(vv)
        hh = Variable(torch.zeros(num_words,100))

        for i in range(self.iterations):
            ww = self.concat(uu, indices)
            bb = self.B.forward(ww, hh, num_words)
            oo, hh = self.B2.forward(bb)
            ll = self.C.forward(oo)
            uu = self.D.forward(hh)

        return ll
