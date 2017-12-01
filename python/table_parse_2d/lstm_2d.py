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

        # Notation:
        #   a = Pre-cell activation
        #   f = Forget gate (y-coordinate)
        #   g = Forget gate (x-coordinate)
        #   k = Input gate
        #   o = Output gate
        #
        #   W = Input weights [input -> hidden]
        #   U = Recurrent weights [ hidden -> hidden] (x-coordinate)
        #   V = Recurrent weights [ hidden -> hidden] (y-coordinate)
        #   b = Bias weight of respective gates

        # Cite: The notation is picked from: https://github.com/jpuigcerver/rnn2d/wiki/LSTM-2D

        self.W_a = Parameter(torch.Tensor(self.D_in, self.D_hidden))
        self.U_a = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.V_a = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.b_a = Parameter(torch.Tensor(self.D_hidden))

        self.W_f = Parameter(torch.Tensor(self.D_in, self.D_hidden))
        self.U_f = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.V_f = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.b_f = Parameter(torch.Tensor(self.D_hidden))

        self.W_g = Parameter(torch.Tensor(self.D_in, self.D_hidden))
        self.U_g = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.V_g = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.b_g = Parameter(torch.Tensor(self.D_hidden))

        self.W_k = Parameter(torch.Tensor(self.D_in, self.D_hidden))
        self.U_k = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.V_k = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.b_k = Parameter(torch.Tensor(self.D_hidden))

        self.W_o = Parameter(torch.Tensor(self.D_in, self.D_hidden))
        self.U_o = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.V_o = Parameter(torch.Tensor(self.D_hidden, self.D_hidden))
        self.b_o = Parameter(torch.Tensor(self.D_hidden))

    def forward(self, x):
        pass