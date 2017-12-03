from table_parse_2d.lstm_2d import ModuleLstm2D
import torch
from torch.autograd import Variable
import unittest




class Tests2D(unittest.TestCase):
    def setUp(self):
        self.x = torch.FloatTensor(10, 256, 256, 308)
        self.x = Variable(self.x).cuda()
        self.lstm2d = ModuleLstm2D(308, 100).cuda()

    def test_lstm_2d(self):
        y = self.lstm2d(self.x)
        y_d = y.data
        batch, height, width, hidden = y_d.size()

        assert batch == 10
        assert height == 256
        assert width == 256
        assert hidden == 200

if __name__ == '__main__':
    unittest.main()
