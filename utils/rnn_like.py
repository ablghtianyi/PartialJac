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
from .partialj_lib_rnn import flatten, fwd_rnn_hook, bwd_rnn_hook
from .layers import Standard_Linear, nnErf

class Recursive_Block(nn.Module):
    def __init__(self, in_size: int, t: int, sigma_w: float, sigma_b: float, act_name: str, bias: bool = True, mu: float = 1.0, norm_type: str ='none') -> None:
        super(Recursive_Block, self).__init__()

        self.in_size = in_size
        self.out_size = self.in_size
        
        self.t = t
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.bias = bias
        self.mu = mu
        self.norm_type = norm_type
        
        if self.act_name == 'ReLU':
            self.act = nn.ReLU()
        elif self.act_name == 'erf':
            self.act = nnErf()
        elif self.act_name == 'GELU':
            self.act = nn.GELU()
        elif self.act_name == 'Linear':
            self.act = nn.Identity()
        elif self.act_name == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise RuntimeError('Error, use only one of the following activation functions : \n \
                ReLU, erf, GELU, Linear, Tanh')

        if self.norm_type == 'none':
            pass
        elif self.norm_type == 'pre' or self.norm_type == 'post':
            self.norm = nn.LayerNorm(self.in_size, bias=self.bias)
        else:
            raise RuntimeError('none, pre or post')
        
        self.linear = Standard_Linear(self.in_size, self.in_size, std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.norm_type == 'none':
            for _ in range(self.t):
                x = self.linear(self.act(x)) + self.mu * x
        elif self.norm_type == 'pre':
            for _ in range(self.t):
                x = self.linear(self.act(self.norm(x))) + self.mu * x
        elif self.norm_type == 'post':
            for _ in range(self.t):
                x = self.norm(self.linear(self.act(x)) + self.mu * x)
        return x
    
    
class Fcc_Shared(nn.Module):
    def __init__(self, in_size: int, h_size: int, out_size: int, n_hidden: int, t: int,\
        sigma_w: float, sigma_b: float, act_name: str, bias: bool = True, mu: float = 1.0, norm_type: str = 'none') -> None:
        super(Fcc_Shared, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        assert n_hidden >= t   # t = n_hidden means complete weight sharing, t = 1 means MLP, control t to interpolate
        assert ((n_hidden // t) * t)  == n_hidden  # Avoid implicit mismatch
        self.n_hidden = n_hidden
        self.t = t
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.bias = bias
        self.mu = mu
        self.norm_type = norm_type
        
        if self.act_name == 'ReLU':
            self.act = nn.ReLU()
        elif self.act_name == 'erf':
            self.act = nnErf()
        elif self.act_name == 'GELU':
            self.act = nn.GELU()
        elif self.act_name == 'Linear':
            self.act = nn.Identity()
        elif self.act_name == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise RuntimeError('Error, use only one of the following activation functions : \n \
                ReLU, erf, GELU, Linear, Tanh')

        self.modlist = nn.ModuleList()

        self.modlist.append(Standard_Linear(self.in_size, self.h_size,
                                            std_weights=1.0, std_bias=self.sigma_b, bias=self.bias))

        for _ in range(self.n_hidden // self.t):
            self.modlist.append(Recursive_Block(self.h_size, t=self.t, sigma_w=self.sigma_w, sigma_b=self.sigma_b, act_name=self.act_name, 
                                                bias=self.bias, mu=self.mu, norm_type=self.norm_type)
                                )
            
        self.modlist.append(self.act)
        self.modlist.append(Standard_Linear(self.h_size, self.out_size,
                                            std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias))
        self.interpolate_init_()
    
    def interpolate_init_(self):
        """
        Copy the data of first Recursive Block to all other blocks, 
        such that we isolate the effect of weight sharing.
        """
        count = 0
        for id, m in self.named_modules():
            if isinstance(m, Recursive_Block):
                if count == 0:
                    ref_block = m
                    continue
                else:
                    m.linear.weight.data = ref_block.linear.weight.data.clone()
                    if self.bias is True:
                        m.linear.bias.data = ref_block.linear.bias.data.clone()
                    
        return
        
    def forward(self, x: Tensor) -> Tensor:
        x = flatten(x)
        for m in self.modlist:
            x = m(x)
        return x
    
    

class RNNLike(nn.Module):
    def __init__(self, in_size: int, h_size: int, out_size: int, n_hidden: int, \
        sigma_w: float, sigma_b: float, act_name: str, bias: bool = True, mu: float = 1.0, norm_type: str = 'none') -> None:
        super(RNNLike, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.n_hidden = n_hidden
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.bias = bias
        self.mu = mu
        self.norm_type = norm_type
        
        if self.act_name == 'ReLU':
            self.act = nn.ReLU()
        elif self.act_name == 'erf':
            self.act = nnErf()
        elif self.act_name == 'GELU':
            self.act = nn.GELU()
        elif self.act_name == 'Linear':
            self.act = nn.Identity()
        elif self.act_name == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise RuntimeError('Error, use only one of the following activation functions : \n \
                ReLU, erf, GELU, Linear, Tanh')

        self.modlist = nn.ModuleList()

        self.input = Standard_Linear(self.in_size, self.h_size,
                                            std_weights=1.0, std_bias=self.sigma_b, bias=self.bias)
        self.modlist.append(self.input)

        for _ in range(self.n_hidden):
            self.modlist.append(Recursive_Block(self.h_size, t=1, sigma_w=self.sigma_w, sigma_b=self.sigma_b, act_name=self.act_name, 
                                                bias=self.bias, mu=self.mu, norm_type=self.norm_type)
                                )
            
        self.modlist.append(self.act)
        self.modlist.append(Standard_Linear(self.h_size, self.out_size,
                                            std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias))
        self.interpolate_init_()
    
    def interpolate_init_(self):
        """
        Copy the data of first Recursive Block to all other blocks, 
        such that we isolate the effect of weight sharing.
        """
        count = 0
        for id, m in self.named_modules():
            if isinstance(m, Recursive_Block):
                if count == 0:
                    ref_block = m
                else:
                    m.linear.weights = ref_block.linear.weights
                    if self.bias is True:
                        m.linear.bias = ref_block.linear.bias
                    if self.norm_type is not 'none':
                        m.norm.weight = ref_block.norm.weight
                        if self.bias is True:
                            m.norm.weight = ref_block.norm.weight
                count += 1
        return
    
        
    def forward(self, x: Tensor) -> Tensor:
        x = flatten(x)
        for m in self.modlist:
            x = m(x)
        return x
    
    
    

class RNN(nn.Module):
    def __init__(self, in_size: int, h_size: int, out_size: int, n_hidden: int, \
        sigma_w: float, sigma_b: float, act_name: str, bias: bool = True, mu: float = 1.0, norm_type: str = 'none') -> None:
        super(RNN, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.n_hidden = n_hidden
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.bias = bias
        self.mu = mu
        self.norm_type = norm_type
        
        if self.act_name == 'ReLU':
            self.act = nn.ReLU()
        elif self.act_name == 'erf':
            self.act = nnErf()
        elif self.act_name == 'GELU':
            self.act = nn.GELU()
        elif self.act_name == 'Linear':
            self.act = nn.Identity()
        elif self.act_name == 'Tanh':
            self.act = nn.Tanh()
        else:
            raise RuntimeError('Error, use only one of the following activation functions : \n \
                ReLU, erf, GELU, Linear, Tanh')

        self.input = Standard_Linear(self.in_size, self.h_size,
                                            std_weights=1.0, std_bias=self.sigma_b, bias=self.bias)
        
        self.rnn = Recursive_Block(self.h_size, t=self.n_hidden, sigma_w=self.sigma_w, sigma_b=self.sigma_b, act_name=self.act_name, 
                                            bias=self.bias, mu=self.mu, norm_type=self.norm_type)
            
        self.head = Standard_Linear(self.h_size, self.out_size,
                                            std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        x = flatten(x)
        x = self.input(x)
        x = self.rnn(x)
        x = self.act(x)
        x = self.head(x)
        return x