import torch
from torch.autograd import Variable

from network.dense import Dense


class ZoneSegment(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, input, output, word_mask):
        pass