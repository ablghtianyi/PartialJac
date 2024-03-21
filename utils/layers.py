# Class used to attach forward/backward hooks.
import math
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Module
from torch import Tensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from copy import deepcopy


class nnErf(nn.Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(nnErf, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return torch.erf(input)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Standard_Linear(nn.Module):
    #   This is adopted from torch Linear class.
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, std_weights: float = 1., std_bias: float = 0., bias: bool = True) -> None:
        super(Standard_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.std_weights = std_weights
        self.std_bias = std_bias

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weights, 0., self.std_weights / math.sqrt(self.weights.size(1)))

        if self.bias is not None:
            init.normal_(self.bias, 0., self.std_bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None:
            return F.linear(input, self.weights, self.bias)
        else:
            scale_bias = 1.
            return F.linear(input, self.weights, scale_bias * self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
