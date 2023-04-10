import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as T

from torch.utils.data import sampler
from torch.utils.data import DataLoader
import torch.optim as optim

import sys
sys.path.append("../utils")
from utils.partialjaclib import *

USE_GPU = True

dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('using device:', device)

def main():
    ########### DATASET ###############
    N_TRAIN = 60000
    N_VAL = 6000
    batch_size = 256

    transform = T.Compose([T.ToTensor()])

    mnist_train = dset.FashionMNIST('./FMNIST', train=True, download=True,
                                    transform=transform)

    mnist_val = dset.FashionMNIST('./FMNIST', train=False, download=True,
                                transform=transform)

    loader_train = DataLoader(mnist_train, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(N_TRAIN)), drop_last=False)

    loader_val = DataLoader(mnist_val, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(N_VAL)))

    loader_fast_train = DataLoader(mnist_train, batch_size=batch_size,
                                sampler=sampler.SubsetRandomSampler(range(5000)))

    loader_fast_val = DataLoader(mnist_val, batch_size=batch_size,
                                sampler=sampler.SubsetRandomSampler(range(500)))

    # model hyper-parameter settings
    wvars = np.arange(0.2, 5.1, 0.2)
    bvars = np.arange(0.0, 2.1, 0.2)
    mus = np.array([0.0, 0.5, 0.9, 1.0])
    width = 500
    depth = 48
    act_name = 'ReLU'          # choose 'ReLU', 'erf' or 'GELU'
    if_bias = True
    fixed_seed = 420

    # training hyper-parameter settings
    losstype = "CSE"
    epochs = 2
    time = 1000000000
    nprint = 5

    # lr search
    lrs = np.geomspace(1e-5, 1., 26)

    train_accs = np.zeros((wvars.shape[0], bvars.shape[0], mus.shape[0]))
    test_accs = np.zeros((wvars.shape[0], bvars.shape[0], mus.shape[0]))
    losses = 100 * np.ones((wvars.shape[0], bvars.shape[0], mus.shape[0]))
    best_lrs = np.min(lrs) * np.ones((wvars.shape[0], bvars.shape[0], mus.shape[0]))

    for i_mu in range(len(mus)):
        for i_w in range(len(wvars)):
            for i_b in range(len(bvars)):
                print('w_var = {0:.2f}, b_var = {1:.2f}, mu = {2:.2f}'.format(wvars[i_w], bvars[i_b], mus[i_mu]))
                for lr in lrs:
                    print('lr =' + str(lr))
                    torch.manual_seed(fixed_seed)
                    model = ResLNFcc(28**2, width, 10, depth, 
                        wvars[i_w]**0.5, bvars[i_b]**0.5, act_name, mus[i_mu], if_bias)
                    
                    optimizer = optim.SGD(model.parameters(), lr, momentum=0, weight_decay=0)
                    
                    train_data = fast_trainer(model, optimizer, epochs = epochs, time = time,
                        loader_train = loader_fast_train, loader_val = loader_fast_val,
                        dtype = dtype, device = device, losstype = losstype)

                    if np.isnan(train_data['loss'][-1])==False and train_data['val_acc'][-1] > test_accs[i_w,i_b,i_mu]:
                        best_lrs[i_w,i_b,i_mu] = lr
                        train_accs[i_w,i_b,i_mu] = train_data['tr_acc'][-1]
                        test_accs[i_w,i_b,i_mu] = train_data['val_acc'][-1]
                        losses[i_w,i_b,i_mu] = train_data['loss'][-1]
                print("Best lr = " + str(best_lrs[i_w,i_b,i_mu]))

    # np.save('./Data/file_name_best_lrs.npy', best_lrs)

    # training loop
    epochs = 10
    time = 1000000000
    nprint = 10

    lrs = best_lrs
    train_accs = np.zeros((wvars.shape[0], bvars.shape[0], mus.shape[0]))
    test_accs = np.zeros((wvars.shape[0], bvars.shape[0], mus.shape[0]))
    losses = np.zeros((wvars.shape[0], bvars.shape[0], mus.shape[0]))

    for i_mu in range(mus.shape[0]):
        for i_w in range(wvars.shape[0]):
            for i_b in range(bvars.shape[0]):
                wvar = wvars[i_w]
                bvar = bvars[i_b]
                mu = mus[i_mu]
                lr = lrs[i_w,i_b,i_mu]
                print('w_var = {0:.2f}, b_var = {1:.2f}, mu = {2:.2f}'.format(wvar, bvar, mu))
                torch.manual_seed(fixed_seed)
                model = ResLNFcc(28**2, width, 10, depth, 
                        wvars[i_w]**0.5, bvars[i_b]**0.5, act_name, mus[i_mu], if_bias)

                optimizer = optim.SGD(model.parameters(), lr, momentum=0, weight_decay=0)

                train_data = trainer(model, optimizer, epochs = epochs, time=time,
                    loader_train = loader_train, loader_val = loader_val,
                    dtype = dtype, device = device, nprint = nprint, losstype = losstype)

                train_accs[i_w, i_b, i_mu] = train_data['tr_acc'][-1]
                test_accs[i_w, i_b, i_mu] = train_data['val_acc'][-1]
                losses[i_w, i_b, i_mu] = train_data['loss'][-1]

    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    losses = np.array(losses)

    # np.save('./Data/file_name_train_accs.npy', train_accs)
    # np.save('./Data/file_nametest_accs.npy', test_accs)
    # np.save('./Data/file_name_losses.npy', losses)

if __name__ == "__main__":
    main()
