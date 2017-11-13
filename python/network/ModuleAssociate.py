import torch
import numpy as np
from torch.autograd import Variable
from network.ModuleC import ModuleC


class ModuleAssociate(torch.nn.Module):
    def __init__(self):
        super(ModuleAssociate, self).__init__()

        # H1 = Variable(torch.randn(num_words, 100))
        # H2 = Variable(torch.randn(num_words, 100))

        self.C1 = ModuleC(100, 100)
        self.C2 = ModuleC(100, 100)

    def forward(self, oo):
        ll = self.C1.forward(oo)
        ll2 = self.C1.forward(oo)

        # ll is Nx100. So if we do llxll' we get NxN as required
        A = torch.mm(ll, ll.transpose(0,1))
        # ll2 is Nx100. So if we do ll2xll2' we get NxN as required
        B = torch.mm(ll2, ll2.transpose(0, 1))
        # Flatten A to N dimensional vector
        A_f = A.view(A.size(0)*A.size(0))
        # Flatten B to N dimensional vector
        B_f = B.view(B.size(0)*B.size(0))
        # Expand dimensions to facilitate concatenation
        A_f = A_f.unsqueeze(1)
        B_f = B_f.unsqueeze(1)
        R = torch.cat((A_f, B_f), dim=1)

        return R

        # return self.linear3(x)