import torch
import numpy as np
from torch.autograd import Variable
from table_detect.dense import Dense

input = Variable(torch.from_numpy(np.zeros((1000, 308)).astype(np.float32)))
config = [300, 'S', 100, 'S', 100, 'T']
module = Dense(308, config)
output = module(input)
size = output.size()

assert size[0] == 1000 and size[1] == 100
