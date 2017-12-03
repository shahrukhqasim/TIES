import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter


class ModuleLstm2D(torch.nn.Module):
    def __init__(self, D_in, D_hidden):
        """
        Remember the shape of input tensor is:
            [N, H, W, D_in]
        and of the output tensor:
            [N, H, W, D_out]
        
        :param D_in: Input dimension
        :param D_hidden: Output dimension
        """

        super(ModuleLstm2D, self).__init__()

        # Store it in the class
        self.D_in = D_in
        self.D_hidden = D_hidden

        self.LSTMx = torch.nn.LSTM(self.D_in, self.D_hidden, 1, batch_first=False, bidirectional=True)
        self.LSTMy = torch.nn.LSTM(self.D_hidden * 2, self.D_hidden, 1, batch_first=False, bidirectional=True)


    def forward(self, x):
        """
        
        Runs the forward pass
        
        x has shape [B, H, W, D_in]
        
        :param x: the Tensor
        :return: 2D LSTM result
        """

        batch, height, width, input_size = x.size()

        """
        We need x in the form of [seq_len, batch, input_size].
        Current:
        [batch, height, width, input_size]
        
        The required sequence length is the width of the image.
        Merge batch and height to get:
        [batch*height, width, input_size]
        Then take the transpose:
        [width, batch*height, input_size]
        It is the required format
        
        """

        x = x.view(batch * height, width, input_size)
        x = x.transpose(0,1).contiguous()

        # Pass through the LSTM
        x, _ = self.LSTMx.forward(x, None)

        """
        x_hidden should be of the size [width, batch * height, hidden_size * 2]
        We need it in the form of [height, batch * width, hidden_size * 2]
        First take transpose to get:
        [batch * height, width, hidden_size * 2]
        Then review as :
        [batch, height, width, hidden_size * 2]
        Then take transpose:
        [height, batch, width, hidden_size * 2]
        Join batch and width:
        [height, batch * width, hidden_size * 2]
        
        """

        x = x.transpose(0, 1).contiguous()
        x = x.view(batch, height, width, self.D_hidden * 2)
        x = x.transpose(0, 1).contiguous()
        x = x.view(height, batch * width, self.D_hidden * 2)
        x, _ = self.LSTMy.forward(x)

        """
        Now it should be of size [height, batch * width, hidden_size * 2]
        Take transpose:
        [batch * width, height, hidden_size]
        View:
        [batch, width, height, hidden_size]
        Transpose:
        [batch, height, width, hidden_size]
        """

        x = x.transpose(0,1).contiguous()
        x = x.view(batch, width, height, self.D_hidden * 2)
        x = x.transpose(1,2).contiguous()


        return x

