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

from utils.rnn_like import Fcc_Shared, RNNLike, Recursive_Block
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
n_hidden = 32
norm_type = sys.argv[1]
act_name = sys.argv[2]
bias = False

# for k, t in enumerate(tqdm(ts)):
#     for n in range(n_ave):
#         torch.manual_seed(seed + n)
#         np.random.seed(seed + n)
#         random.seed(seed + n)
#         for i in range(wvars.shape[0]):
#             for j in range(mus.shape[0]):
#                 model = Fcc_Shared(n_channel * img_size**2, width, out_size, n_hidden, t, wvars[i]**0.5, 0.0, act_name=act_name, bias=bias, mu=mus[j], norm_type=norm_type).to(device=device)
#                 hooks = ghook_rnn(model, batch_size, n_proj, device, cut=(n_hidden // t) - 1)
#                 out = model(xr)
#                 # print(hooks[-1].pj.cpu().numpy(), len(hooks))
#                 chijs[k, i, j] += hooks[-1].pj.cpu().numpy()

# chijs = chijs / n_ave
path = f'{norm_type}_{act_name}_d{n_hidden}_loop.npy'

if os.path.exists(path):
    data = np.load(path, allow_pickle=True).item()
    chijs = data['Js']
    kernels = data['nngps']

else:
    for ave in tqdm(range(n_ave)):
        torch.manual_seed(seed + ave)
        np.random.seed(seed + ave)
        random.seed(seed + ave)
        for i in range(wvars.shape[0]):
            for j in range(mus.shape[0]):
                model = Fcc_Shared(n_channel * img_size**2, width, out_size, n_hidden, n_hidden, wvars[i]**0.5, 0.0, act_name=act_name, bias=bias, mu=mus[j], norm_type=norm_type).to(device=device)
                with torch.inference_mode():
                    h1 = model.input(flatten(xr))
                    k0 = torch.sum(h1**2) / width
                fhooks = fwd_rnn_hooks(model, Recursive_Block, low=1, high=n_hidden)
                bhook = bwd_rnn_hook(model, Recursive_Block, n=0)
                out = model(xr)
                Js, nngps = partialjs_nngps(fhooks, bhook, n_proj, device)
                chijs[:, i, j] += Js
                nngps = np.insert(nngps, 0, k0.item())
                kernels[:, i, j] += nngps[:-1]
                
    chijs = chijs / n_ave
    kernels = kernels / n_ave
    print(chijs[-1])
    print(kernels[-1])
    np.save(path ,{'Js': chijs, 'nngps': kernels})


levels = np.linspace(0.0, 2.0, 200)
fig, axs = plt.subplots(1, len(ts), figsize=(5 * len(ts), 5), constrained_layout=True)

for i, ax in enumerate(axs):
    csf = ax.contourf(wvars, mus, chijs[i].T, cmap='bwr', levels=levels)
    # print(f't={i} \n', chijs[i].T)
    ax.set_title(f't={ts[i]}')
    ax.set_xlabel('$\\sigma_w^2$')
    ax.set_ylabel('$\\mu$')
    ax.grid()
    
fig.colorbar(csf, ax=axs, format='%.1f')
# fig.savefig(f'{norm_type}_{act_name}.pdf', format='pdf', bbox_inches='tight')
fig.suptitle(f'{norm_type} {act_name}')
plt.show()



# wvars = np.arange(0.0, 4.1, 0.5)
# bvars = np.arange(0.0, 2.1, 0.4)
# acts = ['ReLU', 'erf', 'GELU']
# chijs = np.zeros((len(acts), wvars.shape[0], bvars.shape[0]))

# act = 'ReLU'
# width = 500
# out_size = 10
# n_hidden = 10
# n_proj = 5
# n_ave = 3
# bias = True

# for k, act in enumerate(acts):
#     for _ in range(n_ave):
#         for i in range(wvars.shape[0]):
#             for j in range(bvars.shape[0]):
#                 model = Fcc_Shared(n_channel * img_size**2, width, out_size, n_hidden, t, wvars[i]**0.5, bvars[j]**0.5, act, bias=bias, mu=0.9, norm_type='pre').to(device=device)
#                 # print(model)
#                 # exit()
#                 hooks = ghook_rnn(model, batch_size, n_proj, device, cut=(n_hidden // t) - 2)
#                 out = model(xr)
#                 # print(wvars[i],hooks[-1].pj.cpu().numpy(), len(hooks))
#                 chijs[k, i, j] += hooks[-1].pj.cpu().numpy()

# chijs = chijs / n_ave

# levels = np.linspace(0.0, 2.0, 200)
# fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# csf0 = axs[0].contourf(wvars, bvars, chijs[0].T, cmap='bwr', levels=levels)

# axs[0].set_title('ReLU')
# axs[0].set_xlabel('$\\sigma_w^2$')
# axs[0].set_ylabel('$\\sigma_b^2$')

# csf1 = axs[1].contourf(wvars, bvars, chijs[1].T, cmap='bwr', levels=levels)

# axs[1].set_title('erf')
# axs[1].set_xlabel('$\\sigma_w^2$')
# axs[1].set_ylabel('$\\sigma_b^2$')

# csf2 = axs[2].contourf(wvars, bvars, chijs[2].T, cmap='bwr', levels=levels)

# axs[2].set_title('GELU')
# axs[2].set_xlabel('$\\sigma_w^2$')
# axs[2].set_ylabel('$\\sigma_b^2$')

# fig.colorbar(csf0, ax=axs, format='%.1f')
# plt.show()