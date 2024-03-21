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


def ghook_rnn(model, bs, n_proj, device, cut=-1):
    hooks = []
    count = 1
    for id, m in model.named_modules():
        if isinstance(m, Recursive_Block):
            if count > cut:
                hooks.append(GradHook(m, bs, n_proj=n_proj, device=device))
            count += 1
            
    return hooks
    
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

batch_size = 1
n_channel = 1
img_size = 28
xr = torch.normal(0, 1.0, (batch_size, n_channel, img_size, img_size)).to(device=device)
xr = Variable(xr, requires_grad=True)
xr.shape


wvars = np.arange(0.01, 4.1, 0.2)
mus = np.arange(0.0, 1.1, 0.1)
acts = ['ReLU',]
ts = np.arange(0, 31, 1)
chijs = np.zeros((len(ts), wvars.shape[0], mus.shape[0]))
kernels = np.zeros((len(ts), wvars.shape[0], mus.shape[0]))

act = 'ReLU'
width = 500
out_size = 10
n_proj = 5
n_ave = 100
# n_hidden = 4
norm_type = 'pre'
act_name = 'ReLU'
bias = False

path = f'dumbversion_{norm_type}_{act_name}_d{ts.max() + 1}.npy'

if os.path.exists(path):
    data = np.load(path, allow_pickle=True).item()
    chijs = data['Js']
    kernels = data['nngps']
    print(chijs)
    print(kernels)

else:
    for k, n_hidden in enumerate(ts + 1):
        for ave in tqdm(range(n_ave)):
            torch.manual_seed(seed + ave)
            np.random.seed(seed + ave)
            random.seed(seed + ave)
            for i in range(wvars.shape[0]):
                for j in range(mus.shape[0]):
                    model = RNN(n_channel * img_size**2, width, out_size, n_hidden, wvars[i]**0.5, 0.0, act_name=act_name, bias=bias, mu=mus[j], norm_type=norm_type).to(device=device)
                    # h1 = model.input(flatten(xr))
                    # h2 = model.rnn(h1)
                    # print(xr.norm().pow(2) / img_size**2)
                    # print(h1.norm().pow(2)/width, h2.norm().pow(2) / width)
                    # jac = torch.autograd.functional.jacobian(model.rnn.forward, h1)
                    # print((jac[0, :, 0, :]**2).sum() / width)
                    hook = GradHook(model.rnn, bs=batch_size, n_proj=n_proj, device=device)
                    h1 = model.input(flatten(xr))
                    h2 = model.rnn(h1)
                    # out = model(xr)
                    with torch.inference_mode():
                        chijs[k, i, j] += hook.pj.item()
                        # nngps = np.insert(nngps, 0, k0.item())
                        kernels[k, i, j] += (h2.norm().pow(2) / width).item()
                    
    chijs = chijs / n_ave
    kernels = kernels / n_ave
    print(chijs)
    print(kernels)
    np.save(path ,{'Js': chijs, 'nngps': kernels})


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
