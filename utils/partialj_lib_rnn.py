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

######################################################################
#####Hooks#####
######################################################################
class Hook:
    def __init__(self, module: Module, backward: bool = False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module: Module, input: Tensor, output: Tensor):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class GradHook:
    def __init__(self, module: Module, bs: int, n_proj: int, device):
        self.hook = module.register_forward_hook(self.grad_fn)
        self.n_proj = n_proj
        self.bs = bs
        self.device = device
        
    def grad_fn(self, module: Module, input: Tensor, output: Tensor):
        Js = 0.0
        for _ in range(self.n_proj):
            inter = output.reshape(self.bs, -1)
            vs = generate_unit_vectors(inter.shape[0], inter.shape[1]).reshape(-1).to(device=self.device)
            temp = torch.autograd.grad(inter.reshape(-1), input, vs, retain_graph=True, create_graph=False, allow_unused=False)[0]
            Js += torch.sum(temp**2)

        self.pgrad = temp.detach()
        self.pj = Js / self.bs / self.n_proj
        self.nngp = torch.mean(inter**2)

    def close(self):
        self.hook.remove()
        

######################################################################
#####Tools#####
######################################################################
def flatten(x: Tensor) -> Tensor:
    N = x.shape[0]  # read in N, C, H, W
    # "flatten" the C * H * W values into a single vector per image
    return x.view(N, -1)


def generate_unit_vectors(batch_size: int, out_size: int) -> Tensor:
    vs = torch.randn((batch_size, out_size))
    return vs / torch.norm(vs, dim=-1, keepdim=True)


def fwd_rnn_hook(model: Module, block: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, block):
            if i == n:
                return Hook(layer, backward=False)
            else:
                i += 1


def bwd_rnn_hook(model: Module, block: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, block):
            if i == n:
                return Hook(layer, backward=True)
            else:
                i += 1
                

def fwd_rnn_hooks(model: Module, block: Module, low: int = 1, high: int = 2) -> list:
    i = 0
    hooks = []
    for id, layer in list(model.named_modules()):
        if isinstance(layer, block):
            if low <= i < high:
                hooks.append(Hook(layer, backward=False))
            i += 1             
    return hooks

######################################################################
#####Partial Jacobian#####
######################################################################

def partialj_nngp(fhook: Hook, bhook: Hook, n_proj: int, device) -> float:
    J = 0.0
    for _ in range(n_proj):
        inter = fhook.input[0]
        inter = flatten(inter.contiguous())
        vs = generate_unit_vectors(
            inter.shape[0], inter.shape[1]).to(device=device)
        inter[0].backward(vs[0], retain_graph=True)
        temp = bhook.input[0]
        J += torch.sum(temp[0] ** 2)

    return J.detach().cpu().numpy() / n_proj, (torch.sum(inter**2) / inter.size(1)).detach().cpu().numpy()


def partialjs_nngps(fhooks: list, bhook: Hook, n_proj: int, device) -> np.array:
    Js = []
    nngps = []
    for fhook in fhooks:
        J, nngp = partialj_nngp(fhook, bhook, n_proj, device)
        Js.append(J)
        nngps.append(nngp)
    Js = np.asarray(Js)
    return Js, nngps