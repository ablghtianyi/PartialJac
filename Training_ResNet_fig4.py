############# lr search ##############

def find_best_lr(
    lrs, loader_train, loader_test, momentum, wd, t, losstype, search_epochs, model, dtype, device
):
    import copy
    from utils.resnetlib import train_one_epoch
    import torch.optim as optim

    best_acc = 0.
    
    for lr in lrs:
        net = copy.deepcopy(model)
        net.to(device=device)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        for e in range(1, search_epochs+1):
            if e == search_epochs:
                if_data=True
            else:
                if_data=False
            train_data = train_one_epoch(
                net, optimizer, t, loader_train, loader_test, dtype, device, losstype=losstype,
                if_data=if_data, verbose=False
            )     
        if train_data['tr_acc'][-1] >= best_acc:
            best_lr = lr
            best_acc = train_data['tr_acc'][-1] 
            
    return best_lr

def main():

    ############# Setup ##############
    import numpy as np
    import torch
    torch.backends.cudnn.deterministic = True
    import torch.nn as nn
    import copy
    import pickle
    
    SEED = 42

    import torchvision.datasets as dset
    import torchvision.transforms as T
    from torch.utils.data import sampler
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim

    from utils.resnetlib import resnet110, train_one_epoch, seed_everything

    dtype = torch.float32

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print("using device: ", device)
    if device == torch.device('cuda:0'):
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        

    ############# CIFAR10 data ##############
    N_TRAIN = 50000
    N_VAL = 5000
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    seed_everything(SEED)
    
    cifar10_train = dset.CIFAR10('./CIFAR10/', train=True, download=True, transform=transform)
    cifar10_test = dset.CIFAR10('./CIFAR10/', train=False, download=False, transform=transform)

    loader_train = DataLoader(cifar10_train, batch_size=128, sampler=sampler.SubsetRandomSampler(range(N_TRAIN)))
    loader_test = DataLoader(cifar10_test, batch_size=128, sampler=sampler.SubsetRandomSampler(range(N_VAL)))


    ############# Training ##############
    mus = np.linspace(0.0, 1.0, 11)

    search_epochs = 10
    epochs = 50
    data_epochs = np.arange(10,epochs+1,10)
    if 1 not in data_epochs:
        data_epochs = np.insert(data_epochs, 0, 1)
    if epochs not in data_epochs:
        data_epochs = np.append(data_epochs, epochs)
    lrs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    wd = 1e-4
    momentum = 0.9
    t = 10000000
    losstype = 'CSE'
    n_avg = 1

    best_lrs = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []


    ############# Setup ##############

    for i in range(n_avg):
        
        train_l2 = []
        test_l2 = []
        train_acc2 = []
        test_acc2 = []
        best_lr2 = []

        for mu in mus:
            
            print(f'run {i+1}, mu = {mu}')

            seed_everything(SEED+i)
            model = resnet110(use_norm='LN', mu=mu)
            model.to(device=device)
            
            lr = find_best_lr(
                lrs, loader_train, loader_test, momentum, wd, t, losstype,
                search_epochs, model, dtype, device
            )
            best_lr2.append(lr)
            print(f'best lr = {lr}')

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
            train_l = []
            test_l = []
            train_acc = []
            test_acc = []

            for e in range(1, epochs+1):
                if e in data_epochs:
                    if_data = True
                    verbose = True
                else:
                    if_data = False
                    verbose = False
                train_data = train_one_epoch(
                    model, optimizer, t, loader_train, loader_test, dtype, device, losstype=losstype,
                    if_data=if_data, verbose=verbose
                )
                if e in data_epochs:
                    train_l.append(train_data['loss'][-1])
                    test_l.append(train_data['val_loss'][-1])
                    train_acc.append(train_data['tr_acc'][-1])
                    test_acc.append(train_data['val_acc'][-1])
                
            train_l2.append(train_l)
            test_l2.append(test_l)
            train_acc2.append(train_acc)
            test_acc2.append(test_acc)
            
        train_losses.append(train_l2)
        test_losses.append(test_l2)
        train_accs.append(train_acc2)
        test_accs.append(test_acc2)
        best_lrs.append(best_lr2)

        train_losses_np = np.array(train_losses)   
        test_losses_np = np.array(test_losses)
        train_accs_np = np.array(train_accs)
        test_accs_np = np.array(test_accs)
        best_lrs_np = np.array(best_lrs)

        data = {
            'model': 'resnet110ln_v2',
            'dataset': 'CIFAR10',
            'mus': mus,
            'search_epochs': search_epochs,
            'epochs': epochs,
            'data_epochs': data_epochs,
            'lrs': lrs,
            'best_lrs': best_lrs_np,
            'wd': wd,
            'momentum': momentum,
            'losstype': losstype,
            'train_losses': train_losses_np,
            'test_losses': test_losses_np,
            'train_accs': train_accs_np,
            'test_accs': test_accs_np,
            'n_avg': n_avg
        }

        with open(f"./resnet110ln_v2_momentum{momentum}_seed{SEED}.pickle", 'wb') as f:
            pickle.dump(data, f)
    

if __name__ == "__main__":
    main()