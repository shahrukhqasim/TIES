import torch
import numpy as np
from torch.autograd import Variable
from network.ModuleC import ModuleC


class ModuleAssociate2(torch.nn.Module):
    def __init__(self):
        super(ModuleAssociate2, self).__init__()
        self.num_features = 30

        self.P = ModuleC(100,30)
        self.C = ModuleC(60, 2)

    def expand_way_1(self, A, num_words):
        return A.view(1, num_words, self.num_features).expand(num_words, 1, num_words, self.num_features).transpose(0, 2).contiguous().view(
            num_words * num_words, self.num_features)

    def expand_way_2(self, A, num_words):
        return A.expand(num_words, num_words, self.num_features).contiguous().view(num_words * num_words, self.num_features)

    def concat_each(self, A, num_words):
        A1 = self.expand_way_1(A, num_words)
        A2 = self.expand_way_2(A, num_words)
        return torch.cat((A1, A2), dim=1)

    def forward(self, oo, num_words):
        oo = self.P(oo)
        each_concatenated = self.concat_each(oo, num_words)
        return self.C.forward(each_concatenated)

        # return self.linear3(x)
