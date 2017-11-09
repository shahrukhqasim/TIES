import torch
import numpy as np
from torch.autograd import Variable


class ModuleD(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(ModuleD, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, 100)
        self.linear2 = torch.nn.Linear(100, 100)
        self.linear3 = torch.nn.Linear(100, D_out)

    def forward(self, x):
        o1 = self.linear1(x).clamp(min=0)
        o2 = self.linear2(o1).clamp(min=0)
        return self.linear3(o2).clamp(min=0)