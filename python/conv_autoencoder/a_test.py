import torch
from torch import nn
from torch.autograd import Variable

image = Variable(torch.FloatTensor(1, 3, 300, 300))

C = nn.Conv2d(3,5,(5,5))

image2 = C(image)

print(image2.size())