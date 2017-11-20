import torch
import numpy as np
from torch.autograd import Variable


class ModuleA(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(ModuleA, self).__init__()

        # H1 = Variable(torch.randn(num_words, 100))
        # H2 = Variable(torch.randn(num_words, 100))

        self.linear1 = torch.nn.Linear(D_in, 200).cuda()
        self.linear2 = torch.nn.Linear(200, 100).cuda()
        self.linear3 = torch.nn.Linear(100, D_out).cuda()

        # For empirical reasons, this better
        self.linear1.weight.data.uniform_(-30, 30)
        self.linear1.bias.data.uniform_(0, 0)
        self.linear2.weight.data.uniform_(-30, 30)
        self.linear2.bias.data.uniform_(0, 0)
        self.linear3.weight.data.uniform_(-30, 30)
        self.linear3.bias.data.uniform_(0, 0)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        o1 = self.activation(self.linear1(x))
        o2 = self.activation(self.linear2(o1))
        return self.activation(self.linear3(o2))

        # return self.linear1(x).tanh()