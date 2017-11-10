import torch
import numpy as np
from torch.autograd import Variable


class ModuleB2(torch.nn.Module):
    def __init__(self, D_in, D_out_1, D_out_2):
        super(ModuleB2, self).__init__()

        # H1 = Variable(torch.randn(num_words, 100))
        # H2 = Variable(torch.randn(num_words, 100))

        self.linear1 = torch.nn.Linear(D_in, 100).cuda()
        self.linear2 = torch.nn.Linear(100, 100).cuda()
        self.linear3 = torch.nn.Linear(100, D_out_1).cuda()

        #
        # H12 = Variable(torch.randn(num_words, 100))
        # H22 = Variable(torch.randn(num_words, 100))

        self.linear12 = torch.nn.Linear(D_in, 100).cuda()
        self.linear22 = torch.nn.Linear(100, 100).cuda()
        self.linear32 = torch.nn.Linear(100, D_out_2).cuda()

    def forward(self, x):
        o1 = self.linear1(x).clamp(min=0)
        o2 = self.linear2(o1).clamp(min=0)

        o12 = self.linear12(x).clamp(min=0)
        o22 = self.linear22(o12).clamp(min=0)

        return self.linear3(o2).tanh(), self.linear32(o22).tanh()

        # return self.linear1(x).tanh(), self.linear12(x).tanh()