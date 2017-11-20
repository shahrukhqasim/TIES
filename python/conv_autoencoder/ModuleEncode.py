import torch
import numpy as np
from torch.autograd import Variable
from torch import nn

class ModuleEncode(torch.nn.Module):

    def __init__(self):
        super(ModuleEncode, self).__init__()

        #config = [48, 48, 'M', 64, 64, 'M', 128, 128, 128, 'M', 192, 192, 192]
        config = [48, 48, 'M', 48, 48, 'M', 48, 48, 48, 'M', 48, 48, 48]

        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)


    def forward(self, x):
        return self.features(x)