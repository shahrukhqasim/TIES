import torch
import numpy as np
from torch.autograd import Variable
from torch import nn
from conv_autoencoder.ModuleEncode import ModuleEncode
from conv_autoencoder.ModuleDecode import ModuleDecode


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = ModuleEncode()
        self.decoder = ModuleDecode()

    def forward(self, x):
        return self.decoder(self.encoder(x))