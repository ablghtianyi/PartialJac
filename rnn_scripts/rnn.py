import sys
from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random
import torch
import numpy as np

import os
import sys
path = os.path.relpath("..")
sys.path.append(path)

from tqdm.auto import tqdm

from utils.rnn_like import Fcc_Shared, RNN, Recursive_Block
from utils.partialj_lib_rnn import GradHook, fwd_rnn_hooks, bwd_rnn_hook, partialjs_nngps, flatten

set_matplotlib_formats("pdf", "svg")

USE_GPU = True

dtype = torch.float32


if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda:0")
elif USE_GPU == True and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device("cpu")
print("using device:", device)


def jac_fn(fn, inputs):
    jac = torch.autograd.functional.jacobian(fn, inputs)[0, :, 0, :]  # (1, dim, 1, dim)
    J2 = jac @ jac.T
    J4 = J2 @ J2
    return (torch.trace(J2) / J2.size(0)).item(), (torch.trace(J4) / J4.size(0)).item()
    
    
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = 1
n_channel = 1
img_size = 28

xr = torch.normal(0, 1.0, (batch_size, n_channel, img_size, img_size)).to(device=device)
# xr /= (xr.pow(2).sum(dim=(-1, -2, -3)) / n_channel / img_size**2)**0.5
xr = Variable(xr, requires_grad=True)

wvars = np.arange(0.01, 4.1, 0.2)
mus = np.arange(0.0, 1.1, 0.1)
# wvars = np.array([1.0])
# mus = np.array([1.0])
n_proj = 5
width = 500
out_size = 10
n_ave = 1
bias = False

norm_type = 'pre'
act_name = 'Linear'
exact = False

ts = np.arange(0, 31, 1)
J2s = np.zeros((len(ts), n_ave, wvars.shape[0], mus.shape[0]))
J4s = np.zeros((len(ts), n_ave, wvars.shape[0], mus.shape[0]))
kernels = np.zeros((len(ts), n_ave, wvars.shape[0], mus.shape[0]))


path = f'../data/{exact}_{norm_type}_{act_name}_d{ts.max() + 2}.npy'

if os.path.exists(path):
    data = np.load(path, allow_pickle=True).item()
    J2s = data['J2s']
    kernels = data['nngps']
    print(J2s)
    print(kernels)

else:
    if exact is False:
        for k, n_hidden in enumerate(ts + 1):
            for ave in tqdm(range(n_ave)):
                torch.manual_seed(seed + ave)
                np.random.seed(seed + ave)
                random.seed(seed + ave)
                for i in range(wvars.shape[0]):
                    for j in range(mus.shape[0]):
                        model = RNN(n_channel * img_size**2, width, out_size, n_hidden, wvars[i]**0.5, 0.0, act_name=act_name, bias=bias, mu=mus[j], norm_type=norm_type).to(device=device)
                        hook = GradHook(model.rnn, bs=batch_size, n_proj=n_proj, device=device)
                        h1 = model.input(flatten(xr))
                        h2 = model.rnn(h1)
                        print(hook.pj.item())
                        with torch.inference_mode():
                            J2s[k, ave, i, j] += hook.pj.item()
                            kernels[k, ave, i, j] += (h2.norm().pow(2) / width).item()
                        
        J2s = J2s
        kernels = kernels
        print(J2s.mean(1))
        print(kernels.mean(1))
        np.save(path ,{'J2s': J2s, 'nngps': kernels})
    else:
        for k, n_hidden in enumerate(ts + 1):
            for ave in tqdm(range(n_ave)):
                torch.manual_seed(seed + ave)
                np.random.seed(seed + ave)
                random.seed(seed + ave)
                for i in range(wvars.shape[0]):
                    for j in range(mus.shape[0]):
                        model = RNN(n_channel * img_size**2, width, out_size, n_hidden, wvars[i]**0.5, 0.0, act_name=act_name, bias=bias, mu=mus[j], norm_type=norm_type).to(device=device)
                        h1 = model.input(flatten(xr))
                        J2, J4 = jac_fn(model.rnn, h1)
                        print(J2, J4)
                        h2 = model.rnn(h1)
                        with torch.inference_mode():
                            J2s[k, ave, i, j] += J2
                            J4s[k, ave, i, j] += J4
                            kernels[k, ave, i, j] += (h2.norm().pow(2) / width).item()
                        
        J2s = J2s / n_ave
        J4s = J4s / n_ave
        kernels = kernels
        print(J2s.mean(1))
        print(J4s.mean(1))
        print(kernels.mean(1))
        np.save(path ,{'J2s': J2s, 'J4s': J4s, 'nngps': kernels})


# levels = np.linspace(0.0, 2.0, 200)
# fig, axs = plt.subplots(1, len(ts), figsize=(5 * len(ts), 5), constrained_layout=True)

# for i, ax in enumerate(axs):
#     csf = ax.contourf(wvars, mus, chijs[i].T, cmap='bwr', levels=levels)
#     # print(f't={i} \n', chijs[i].T)
#     ax.set_title(f't={ts[i]}')
#     ax.set_xlabel('$\\sigma_w^2$')
#     ax.set_ylabel('$\\mu$')
#     ax.grid()
    
# fig.colorbar(csf, ax=axs, format='%.1f')
# # fig.savefig(f'{norm_type}_{act_name}.pdf', format='pdf', bbox_inches='tight')
# fig.suptitle(f'{norm_type} {act_name}')
# plt.show()
