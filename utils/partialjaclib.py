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
#####Models#####
######################################################################
class Fcc_Standard(nn.Module):
    def __init__(self, in_size: int, h_size: int, out_size: int, n_hidden: int, \
        sigma_w: float, sigma_b: float, act_name: str, bias: bool = True, p: float = 0, b: int = 1e4) -> None:
        super(Fcc_Standard, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.n_hidden = n_hidden
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.bias = bias
        self.p = p
        self.b = b
        
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

        for _ in range(self.n_hidden):
            # self.modlist.append(nn.LayerNorm(
            #     self.h_size, elementwise_affine=False))
            self.modlist.append(self.act)
            self.modlist.append(Standard_Linear(
                self.h_size, self.h_size, std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias))

        # self.modlist.append(nn.LayerNorm(
        #     self.h_size, elementwise_affine=False))
        self.modlist.append(self.act)
        # self.modlist.append(nn.LayerNorm(self.h_size, elementwise_affine=False))
        self.modlist.append(Standard_Linear(self.h_size, self.out_size,
                            std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias))

    def forward(self, x: Tensor) -> Tensor:
        x = flatten(x)
        for m in self.modlist:
            x = m(x)
        return x


class ResFcc(nn.Module):
    def __init__(self, in_size: int, h_size: int, out_size: int, n_hidden: int, 
                 sigma_w: float, sigma_b: float, act_name: str, mu: float = 1.0, bias: bool = True) -> None:
        super(ResFcc, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.n_hidden = n_hidden
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.mu = mu
        self.bias = bias

        if self.act_name == 'ReLU':
            self.act = nn.ReLU()
        elif self.act_name == 'erf':
            self.act = nnErf()
        elif self.act_name == 'GELU':
            self.act = nn.GELU()
        elif self.act_name == 'Linear':
            self.act = nn.Identity()
        else:
            raise RuntimeError('Error: use only ReLU, erf, GELU or Linear')
        
        self.modlist = nn.ModuleList()

        self.modlist.append(Standard_Linear(self.in_size, self.h_size,
                            std_weights=1.0, std_bias=self.sigma_b, bias=self.bias))

        for _ in range(self.n_hidden):
            self.modlist.append(ResBlock(self.h_size, self.h_size, self.sigma_w,
                                self.sigma_b, self.act_name, mu=self.mu, bias=self.bias))

        self.modlist.append(self.act)
        self.modlist.append(Standard_Linear(self.h_size, self.out_size,
                            std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias))

    def forward(self, x: Tensor) -> Tensor:
        x = flatten(x)
        for m in self.modlist:
            x = m(x)
        return x
    

class ResLNFcc(nn.Module):
    def __init__(self, in_size: int, h_size: int, out_size: int, n_hidden: int, 
                 sigma_w: float, sigma_b: float, act_name: str, mu: float = 1.0, bias: bool = True) -> None:
        super(ResLNFcc, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.n_hidden = n_hidden
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.mu = mu
        self.bias = bias
        
        if self.act_name == 'ReLU':
            self.act = nn.ReLU()
        elif self.act_name == 'erf':
            self.act = nnErf()
        elif self.act_name == 'GELU':
            self.act = nn.GELU()
        elif self.act_name == 'Linear':
            self.act = nn.Identity()
        else:
            raise RuntimeError('Error: use only ReLU, erf, GELU or Linear')

        self.modlist = nn.ModuleList()

        self.modlist.append(Standard_Linear(self.in_size, self.h_size,
                            std_weights=1.0, std_bias=self.sigma_b, bias=self.bias))

        for _ in range(self.n_hidden):
            self.modlist.append(ResLNBlock(self.h_size, self.h_size, self.sigma_w,
                                self.sigma_b, self.act_name, mu=self.mu, bias=self.bias))

        self.modlist.append(self.act)
        self.modlist.append(Standard_Linear(self.h_size, self.out_size,
                            std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias))

    def forward(self, x: Tensor) -> Tensor:
        x = flatten(x)
        for m in self.modlist:
            x = m(x)
        return x


class MlpMixer(nn.Module):
    def __init__(self, img_size: int, patch_size: int, h_size: int, tokens_mlp_dim: int, channels_mlp_dim: int, n_classes: int, n_blocks: int, sigma_w: float, sigma_b: float, mu: float, act_name: str, bias: bool = True) -> None:
        super(MlpMixer, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.h_size = h_size
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.n_classes = n_classes
        self.n_blocks = n_blocks

        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.mu = mu
        self.act_name = act_name
        self.bias = bias

        self.n_patches = (img_size // self.patch_size) ** 2

        self.stem = nn.Conv2d(in_channels=3, out_channels=self.h_size,
                              kernel_size=patch_size, stride=patch_size)

        self.modlist = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.modlist.append(MixerBlock(self.n_patches, self.h_size, self.tokens_mlp_dim,
                                self.channels_mlp_dim, self.mu, self.act_name, bias=self.bias))

        self.pre_head_ln = nn.LayerNorm(self.h_size, elementwise_affine=True)
        self.head = nn.Linear(self.h_size, self.n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.0, mode='fan_in', nonlinearity='linear')
                init.normal_(m.bias, 0.0, 0.0)
            if isinstance(m, nn.Linear):
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                init.normal_(m.weight, 0.0, self.sigma_w / math.sqrt(float(fan_in)))
                init.normal_(m.bias, 0.0, self.sigma_b)


    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        for block in self.modlist:
            x = block(x)
        x = self.pre_head_ln(x)
        x = torch.mean(x, 1)
        x = self.head(x)
        return x


######################################################################
#####Layers#####
######################################################################
class MlpBlock(Module):
    def __init__(self, h_size: int, inter_size: int, act_name: str, bias: bool = True) -> None:
        super(MlpBlock, self).__init__()

        self.h_size = h_size
        self.inter_size = inter_size
        self.act_name = act_name
        self.bias = bias

        self.modlist = nn.ModuleList()

        self.modlist.append(nn.Linear(self.h_size, self.inter_size))

        if self.act_name == "ReLU":
            self.modlist.append(nn.ReLU())
        elif self.act_name == "erf":
            self.modlist.append(nnErf())
        elif self.act_name == "GELU":
            self.modlist.append(nn.GELU())
        else:
            raise RuntimeError("Error, use only ReLU, erf and GELU")

        self.modlist.append(nn.Linear(self.inter_size, self.h_size))

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modlist:
            x = m(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, n_patches: int, h_size: int, tokens_mlp_dim: int, channels_mlp_dim: int, mu: float, act_name: str, bias: bool = True) -> None:
        super(MixerBlock, self).__init__()

        self.n_patches = n_patches
        self.h_size = h_size
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.mu = mu
        self.act_name = act_name
        self.bias = bias

        self.norm1 = nn.LayerNorm(self.h_size, elementwise_affine=True)
        self.block1 = MlpBlock(self.n_patches, self.tokens_mlp_dim, self.act_name, bias=self.bias)

        self.norm2 = nn.LayerNorm(self.h_size, elementwise_affine=True)
        self.block2 = MlpBlock(self.h_size, self.channels_mlp_dim, self.act_name, bias=self.bias)

    def forward(self, x: Tensor) -> Tensor:
        id = x
        x = self.norm1(x)
        x = x.transpose(-1, -2)
        x = self.block1(x)
        x = x.transpose(-1, -2)
        x = x + self.mu * id

        id = x
        x = self.norm2(x)
        x = self.block2(x)
        x = x + self.mu * id
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_size: int, 
        out_size: int, 
        sigma_w: float, 
        sigma_b: float, 
        act_name: str, 
        mu: float, 
        bias: bool = True,
    ) -> None:
        super(ResBlock, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.mu = mu
        self.bias = bias

        self.fc = Standard_Linear(self.in_size, self.out_size,
                              std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias)
        
        if self.act_name == 'ReLU':
            self.act = nn.ReLU()
        elif self.act_name == 'erf':
            self.act = nnErf()
        elif self.act_name == 'GELU':
            self.act = nn.GELU()
        elif self.act_name == 'Linear':
            self.act = nn.Identity()
        else:
            raise RuntimeError('Error: use only ReLU, erf, GELU or Linear')

    def forward(self, x: Tensor) -> Tensor:
        x = self.mu * x + self.fc(self.act(x))
        return x
    
    
class ResLNBlock(nn.Module):
    def __init__(
        self,
        in_size: int, 
        out_size: int, 
        sigma_w: float, 
        sigma_b: float, 
        act_name: str, 
        mu: float, 
        bias: bool = True,
    ) -> None:
        super(ResLNBlock, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.act_name = act_name
        self.mu = mu
        self.bias = bias

        self.fc = Standard_Linear(self.in_size, self.out_size,
                              std_weights=self.sigma_w, std_bias=self.sigma_b, bias=self.bias)
        
        if self.act_name == 'ReLU':
            self.act = nn.ReLU()
        elif self.act_name == 'erf':
            self.act = nnErf()
        elif self.act_name == 'GELU':
            self.act = nn.GELU()
        elif self.act_name == 'Linear':
            self.act = nn.Identity()
        else:
            raise RuntimeError('Error: use only ReLU, erf, GELU or Linear')

        self.norm = nn.LayerNorm(self.in_size, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mu * x + self.fc(self.act(self.norm(x)))
        return x


# This class is Module Erf layer.
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


######################################################################
#####Functions#####
######################################################################
# Single Layer/Block Hooks
def fwd_hook(model: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, Standard_Linear):
            if i == n:
                return Hook(layer, backward=False)
            else:
                i += 1


def bwd_hook(model: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, Standard_Linear):
            if i == n:
                return Hook(layer, backward=True)
            else:
                i += 1


def resblock_fwd_hook(model: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, ResBlock):
            if i == n:
                return Hook(layer, backward=False)
            else:
                i += 1


def resblock_bwd_hook(model: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, ResBlock):
            if i == n:
                return Hook(layer, backward=True)
            else:
                i += 1
                
                
def reslnblock_fwd_hook(model: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, ResLNBlock):
            if i == n:
                return Hook(layer, backward=False)
            else:
                i += 1


def reslnblock_bwd_hook(model: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, ResLNBlock):
            if i == n:
                return Hook(layer, backward=True)
            else:
                i += 1


def mixerblock_fwd_hook(model: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, MixerBlock):
            if i == n:
                return Hook(layer, backward=False)
            else:
                i += 1


def mixerblock_bwd_hook(model: Module, n: int = 0) -> Hook:
    i = 0
    for id, layer in list(model.named_modules()):
        if isinstance(layer, MixerBlock):
            if i == n:
                return Hook(layer, backward=True)
            else:
                i += 1


######################################################################
#####Partial Jacobian#####
######################################################################

def partialj(fhook: Hook, bhook: Hook, n_proj: int, device) -> float:
    J = 0.0
    for _ in range(n_proj):
        inter = fhook.input[0]
        inter = flatten(inter.contiguous())
        vs = generate_unit_vectors(
            inter.shape[0], inter.shape[1]).to(device=device)
        inter[0].backward(vs[0], retain_graph=True)
        temp = bhook.input[0]
        J += torch.sum(temp[0] ** 2)

    return J.detach().cpu().numpy() / n_proj


def partialjs(fhooks: list, bhook: Hook, n_proj: int, device) -> np.array:
    Js = []
    for fhook in fhooks:
        J = partialj(fhook, bhook, n_proj, device)
        Js.append(J)
    Js = np.asarray(Js)
    return Js

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


@torch.no_grad()
def get_traindict(model: nn.Module, lr: float, decay: float = 0.0):
    all_params = tuple([p for p in model.parameters() if p.requires_grad==True])
    wd_params = list()
    no_wd_params = list()
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            wd_params.append(m.weight)
            no_wd_params.append(m.bias)
        if isinstance(m, (nn.LayerNorm)):
            no_wd_params.append(m.weight)
            no_wd_params.append(m.bias)
        
    # Only weights of specific layers should undergo weight decay.
    # no_wd_params = [p for p in all_params if p not in wd_params]
    print(len(wd_params), len(no_wd_params), len(all_params))
    assert len(wd_params) + len(no_wd_params) == len(all_params), "Sanity check failed."
    
    lrs_wd = []
    lrs_nowd = []
    for i in range(len(wd_params)):
        lrs_wd.append(lr * np.exp(- decay * i))
    
    for i in range(len(no_wd_params)):
        lrs_nowd.append(lr * np.exp(- decay * i))
    
    pm_group = []
    
    for pm, lr in zip(wd_params, lrs_wd):
        pm_group.append({'params': [pm], 'lr': lr})
    
    for pm, lr in zip(no_wd_params, lrs_nowd):
        pm_group.append({'params': [pm], 'lr': lr, 'weight_decay': 0.0})
    
    return pm_group


######################################################################
#####Training#####
######################################################################
def range_fn(min_lr: float = 1e-4, max_lr: float = 10, nums: int = 50) -> np.array:
    return np.geomspace(min_lr, max_lr, nums)


def lrs_grid(wvars: np.array, bvars: np.array, nums: int = 50) -> np.array:
    lrs = np.ones((wvars.shape[-1], bvars.shape[-1], nums))
    # for i in range(wvars.shape[-1]):
    #     for j in range(bvars,shape[-1]):
    #         lrs[i, j, :] = range_fn(nums=nums)
    return lrs * range_fn(nums=nums)


def check_accuracy(loader: DataLoader, model: Module, dtype, device: str) -> tuple:
    num_correct = 0
    num_samples = 0
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    x_wrong = []
    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, preds = scores.max(1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            x_wrong.append(x[y != preds])

        acc = float(num_correct) / num_samples

    return num_correct, num_samples, acc


def check_val_accuracy(
    loader: DataLoader, model: Module, dtype, device: str, criterion
) -> tuple:
    num_correct = 0
    num_samples = 0
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    x_wrong = []
    with torch.no_grad():
        val_loss = 0.0
        for x, y in loader:

            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, preds = scores.max(1)

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            x_wrong.append(x[y != preds])
            val_loss += criterion(scores, y).item()

        acc = float(num_correct) / num_samples

    return num_correct, num_samples, acc, val_loss / len(loader)


def trainer(
    model: Module, 
    optimizer, 
    epochs: int, 
    time: int, 
    loader_train: DataLoader, 
    loader_val: DataLoader, 
    dtype, 
    device: str, 
    nprint: int, 
    losstype: str = 'MSE',
    if_save: bool = False,
    save_epoch_list : list = [],
    save_location: str = './',
    if_state_dict = False,
    state_dict_every = 100
) -> list:
    # Make data dic, contains training data
    data = {'tr_acc': [], 'val_acc': [], 'loss': [],
            'jac': [], 'grad': [], 'grad0': [], 'gradf': [],
            'state_dict_epoch': [], 'model_state_dict': [],
            'optimizer_state_dict': [], 'scheduler_state_dict': []}
    
    if save_location[-1] != '/':
        save_location = save_location + '/'

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    
    if if_save == True:
        torch.save(
            {
                'model' : model,
                'optimizer' : optimizer,
                # 'scheduler' : scheduler
            },
            save_location + 'init.pt'
        )
    
    if if_state_dict == True:
        data['state_dict_epoch'].append(0)
        data['model_state_dict'].append(deepcopy(model.state_dict()))
        data['optimizer_state_dict'].append(deepcopy(optimizer.state_dict()))
        # data['scheduler_state_dict'].append(deepcopy(scheduler.state_dict()))
        
    if losstype == 'MSE':
        criterion = nn.MSELoss()
    elif losstype == 'CSE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError('Choose only MSE or CSE!')

    stopwatch = 0
    for e in range(epochs):
        if e % nprint == 0 or e == epochs - 1:
            print("EPOCH: ", e+1)

        for t, (x, y) in enumerate(loader_train):

            if stopwatch == time:
                return data

            stopwatch += 1
            model.train()  # put model to training mode
            model = model.to(device=device)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            y = y.to(dtype=torch.long)

            scores = model(x).squeeze()
            if losstype == 'MSE':
                loss = criterion(
                    scores, 
                    F.one_hot(y, num_classes=10).to(device=device, dtype=dtype)
                )
            elif losstype == 'CSE':
                loss = criterion(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            data['loss'].append(loss.item())

        if (e+1) % nprint == 0 or (e+1) == 1 or (e+1) == epochs - 1:
            print('Iteration %d, loss = %.4f' % ((t+1)*(e+1), loss.item()))

            num_correct, num_samples, tr_acc = check_accuracy(
                loader_train, model, dtype, device)
            data['tr_acc'].append(tr_acc)

            print('TRAIN : Got %d / %d correct (%.2f)' %
                  (num_correct, num_samples, 100 * tr_acc))
            
            num_correct, num_samples, val_acc = check_accuracy(
                loader_val, model, dtype, device)
            data['val_acc'].append(val_acc)

            print('TEST : Got %d / %d correct (%.2f)' %
                  (num_correct, num_samples, 100 * val_acc))
        
        if if_save == True:
            if e+1 in save_epoch_list or e+1 == epochs:
                torch.save(
                    {
                        'epoch' : e+1,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        # 'scheduler_state_dict' : scheduler.state_dict()
                    },
                    save_location + 'state_dicts_' + str(e+1) + '.pt'
                )
            
        if if_state_dict == True:
            if (e+1) % state_dict_every == 0 or (e+1) == epochs:
                data['state_dict_epoch'].append(e+1)
                data['model_state_dict'].append(deepcopy(model.state_dict()))
                data['optimizer_state_dict'].append(deepcopy(optimizer.state_dict()))

    return data


def fast_trainer(
    model: Module,
    optimizer,
    epochs: int,
    time: int,
    loader_train: DataLoader,
    loader_val: DataLoader,
    dtype,
    device: str,
    losstype: str = "MSE",
) -> list:
    # Make data dic, contains training data
    data = {
        "tr_acc": [],
        "val_acc": [],
        "loss": [],
        "val_loss": [],
        "jac": [],
        "grad": [],
        "grad0": [],
        "gradf": [],
    }

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    if losstype == "MSE":
        criterion = nn.MSELoss()
    elif losstype == "CSE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError("Choose only MSE or CSE!")

    stopwatch = 0
    for e in range(epochs):
        print("EPOCH: ", e)

        for t, (x, y) in enumerate(loader_train):

            if stopwatch == time:
                return data

            stopwatch += 1
            model.train()  # put model to training mode
            model = model.to(device=device)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x).squeeze()
            if losstype == 'MSE':
                loss = criterion(
                    scores, 
                    F.one_hot(y, num_classes=10).to(device=device, dtype=dtype)
                )
            elif losstype == 'CSE':
                loss = criterion(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if e == epochs-1:
            data["loss"].append(loss.item())
            print("Iteration %d, loss = %.4f" % ((t + 1) * (e + 1), loss.item()))
            num_correct, num_samples, running_val = check_accuracy(loader_train, model, dtype, device)
            data['tr_acc'].append(running_val)

            print('TRAIN : Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * running_val ))

            num_correct, num_samples, running_val = check_accuracy(loader_val, model, dtype, device)
            data['val_acc'].append(running_val)

            print('TEST : Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * running_val ))

    return data


def mixup_data(x, y, device, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def trainer_mixup(
        model: Module, num_classes: int, optimizer, scheduler, epochs: int, time: int, loader_train: DataLoader, loader_val: DataLoader, dtype, device: str, nprint: int, alpha: float, warmup: bool = False, path: str = '.') -> list:
    data = {"tr_acc": [], "val_acc": [], "loss": [], "val_loss": []}

    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    stopwatch = 0
    scaler = torch.cuda.amp.GradScaler()
    lr_max = optimizer.param_groups[0]['lr']
    max_accuracy = 0.0
    for e in range(epochs):
        print("EPOCH: ", e)

        for t, (x, y) in enumerate(loader_train):
            T = (t + e * x.shape[0])

            if stopwatch == time:
                return data

            if warmup == True and T < 1000:
                for group in optimizer.param_groups:
                    group['lr'] = (T + 1) / 1000 * lr_max

            x, y = x.to(device=device, dtype=dtype), y.to(
                device=device, dtype=torch.long
            )

            x, y_a, y_b, lam = mixup_data(x, y, device, alpha=alpha)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))

            stopwatch += 1
            model.train()
            with torch.cuda.amp.autocast():
                scores = model(x).squeeze()
                loss = mixup_criterion(criterion, scores, y_a, y_b, lam)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        data["loss"].append(loss.item())
        print("Iteration %d, loss = %.4f" %
              ((t + 1) * (e + 1), loss.item()))

        if e % nprint == 0 or e == epochs - 1:
            num_correct, num_samples, running_val = check_accuracy(
                loader_train, model, dtype, device
            )
            data["tr_acc"].append(running_val)

            print(
                "TRAIN : Got %d / %d correct (%.2f)"
                % (num_correct, num_samples, 100 * running_val)
            )

            num_correct, num_samples, running_val, val_loss = check_val_accuracy(
                loader_val, model, dtype, device, criterion
            )
            data["val_acc"].append(running_val)
            data["val_loss"].append(val_loss)

            print(
                "TEST : Got %d / %d correct (%.2f)"
                % (num_correct, num_samples, 100 * running_val)
            )
            print("Iteration %d, val_loss = %.4f" %
                  ((t + 1) * (e + 1), val_loss))

            print('lr={:e}'.format(optimizer.param_groups[0]['lr']))

        if scheduler is not None:
            scheduler.step()
        
        if max_accuracy < max(data['val_acc']):
            max_accuracy = max(data['val_acc'])
            
            checkpoint_paths = [path + 'best_checkpoint_' + '.pt']
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'epoch': e,
                    'scaler': scaler.state_dict(),
                }, checkpoint_path)
            
            np.save(checkpoint_path, data)

    return data, model
