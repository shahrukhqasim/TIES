import torch
import numpy as np
from torch.autograd import Variable


class ModuleC(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(ModuleC, self).__init__()

        # H1 = Variable(torch.randn(num_words, 100))
        # H2 = Variable(torch.randn(num_words, 100))

        self.linear1 = torch.nn.Linear(D_in, 100)
        self.linear2 = torch.nn.Linear(100, 100)
        self.linear3 = torch.nn.Linear(100, D_out)

    def forward(self, x):
        o1 = self.linear1(x).clamp(min=0)
        o2 = self.linear2(o1).clamp(min=0)
        return self.linear3(o2).clamp(min=0)