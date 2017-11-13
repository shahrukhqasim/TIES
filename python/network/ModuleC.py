import torch
import numpy as np
from torch.autograd import Variable


class ModuleC(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(ModuleC, self).__init__()

        # H1 = Variable(torch.randn(num_words, 100))
        # H2 = Variable(torch.randn(num_words, 100))

        self.linear3 = torch.nn.Linear(D_in, D_out).cuda()

    def forward(self, x):
        return self.linear3(x)

        # return self.linear3(x)