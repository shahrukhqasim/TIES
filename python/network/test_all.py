import unittest
from network.ModuleA import ModuleA
from network.ModuleB import ModuleB
from network.ModuleB2 import ModuleB2
from network.ModuleC import ModuleC
from network.ModuleD import ModuleD
from network.ModuleCollect import ModuleCollect
import numpy as np
import torch
from torch.autograd import Variable
from network.computation_graph import SimpleDocProcModel


class NetsTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def test_model_1(self):
        N = 500
        indices_test = (np.random.randint(0,N, (N,4))).astype(np.int32)
        vv = Variable(torch.randn(N, 330))
        model = SimpleDocProcModel()
        model.set_iterations(20)
        ll = model.forward(indices_test, vv)

if __name__ == '__main__':
    unittest.main()
