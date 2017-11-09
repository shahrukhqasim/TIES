import torch
import numpy as np
from torch.autograd import Variable


class ModuleB(torch.nn.Module):
    def __init__(self):
        super(ModuleB, self).__init__()
        self.gru = torch.nn.GRUCell(500, 100)

    def forward(self, x, hx):
        return self.gru.forward(x, hx)