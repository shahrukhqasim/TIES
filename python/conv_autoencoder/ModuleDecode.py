import torch
import numpy as np
from torch.autograd import Variable
from torch import nn

class ModuleDecode(torch.nn.Module):

    def __init__(self):
        super(ModuleDecode, self).__init__()

        #config = [3, 48, 48, 'M', 64, 64, 'M', 128, 128, 128, 'M', 192, 192]
        config = [3, 48, 48, 'M', 48, 48, 'M', 48, 48, 48, 'M', 48, 48]
        config.reverse()

        layers = []
        # in_channels = 192
        in_channels = 48
        for v in config:
            if v == 'M':
                layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.image = nn.Sequential(*layers)


    def forward(self, x):
        # o1 = self.linear1(x).clamp(min=0)
        # o2 = self.linear2(o1).clamp(min=0)
        return self.image(x)

        # return self.linear3(x).tanh()