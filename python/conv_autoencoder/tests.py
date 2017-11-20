import unittest
import torch
from conv_autoencoder.ModuleEncode import ModuleEncode
from conv_autoencoder.ModuleDecode import ModuleDecode
from torch.autograd import Variable


class NetsTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.encoder = ModuleEncode()
        self.decoder = ModuleDecode()

    def test_encode_decode(self):
        test_random_image = torch.FloatTensor(1, 3, 512, 512)
        image_variable = Variable(test_random_image)
        output_features = self.encoder(image_variable)
        print("Encoded size", output_features.size())
        image = self.decoder(output_features)
        print("Decoded size", image.size())


if __name__ == '__main__':
    unittest.main()
