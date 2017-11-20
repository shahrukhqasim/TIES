import torch
import numpy as np
from torch.autograd import Variable


class Dense(torch.nn.Module):
    def __init__(self, D_in, config = [300,'S',100,'S',100,'T']):
        super(Dense, self).__init__()
        layers = []
        last = D_in
        for v in config:
            if v=='R':
                layers += [torch.nn.ReLU()]
            elif v=='S':
                layers += [torch.nn.Sigmoid()]
            elif v=='T':
                layers += [torch.nn.Tanh()]
            else:
                num_next = int(v)
                layers += [torch.nn.Linear(last, num_next)]
                last = num_next

        self.output = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.output(x)