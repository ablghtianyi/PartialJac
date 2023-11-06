import torch
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import sampler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import sys
sys.path.append("../utils")
from utils.partialjaclib import *


def main():
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("using device:", device)
    N_TRAIN = 50000
    N_VAL = 10000
    batch_size = 256
    # CIFAR10 with Data Augmentation
    transform = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandAugment(
    ), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    transform2 = T.Compose([T.ToTensor(), T.Normalize(
        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    cifar10_train = CIFAR10("./cifar10", train=True,
                            download=True, transform=transform)

    cifar10_val = CIFAR10("./cifar10", train=False,
                            download=True, transform=transform2)

    loader_train = DataLoader(
        cifar10_train,
        batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(range(N_TRAIN)),
        drop_last=False,
        pin_memory=True,
        num_workers=2
    )

    loader_val = DataLoader(
        cifar10_val,
        batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(range(N_VAL)),
        pin_memory=True,
        num_workers=2
    )

    # Training
    t = 1000000000
    epochs = 600
    nprint = 10

    act = "GELU"
    sigma_w = 2.0 ** 0.5
    sigma_b = 0.0 ** 0.5

    img_size = 32
    patch_size = 4
    h_size = 128
    tokens_mlp_dim = 256
    channels_mlp_dim = 256
    n_classes = 10
    n_blocks = 100

    mu = 1.0
    bias = True
    warmup = False
    nran = "run1_"
    train_data_list = []
    i = 0
    shared_name = (
        "d100LN_cifar10_cse_myscheduler_warmup_mixup_augflip_mu1_b0_w2_"
        + act
        + "_wd1e4_"
        + str(epochs)
        + "epochs_"
        + nran
    )

    lrs = [0.5]
    for lr in lrs:
        save_path = "./Data/" + shared_name + str(lr)[:5]
        model = MlpMixer(img_size, patch_size, h_size, tokens_mlp_dim, channels_mlp_dim,
                            n_classes, n_blocks, sigma_w, sigma_b, mu, act, bias=bias)
        
        # for m in model.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         print(m.weight.var() * init._calculate_fan_in_and_fan_out(m.weight)[0])
                
        total_params = sum(p.numel()
                            for p in model.parameters() if p.requires_grad)
        print("Total Parameters: ", total_params)

        print("sigma_w is {:.2f}, sigma_b is {:.2f}, lr is {:.6f}".format(
            sigma_w, sigma_b, lr))

        pm_group = get_traindict(model, lr)
        optimizer = optim.SGD(pm_group, lr,
                                momentum=0.0, weight_decay=1e-4)
        
        scheduler = MultiStepLR(optimizer, milestones=[(
            3 * epochs) // 4, (9 * epochs) // 10], gamma=0.1)

        train_data, model = trainer_mixup(model, n_classes, optimizer, scheduler, epochs,
                                            t, loader_train, loader_val, dtype, device, nprint, 0.8, warmup=warmup, path=save_path)

        np.save(
            save_path + '.npy', train_data
        )
        torch.save(
            model.state_dict(),
            save_path + ".pt",
        )

        train_data_list.append(train_data)


if __name__ == "__main__":
    main()
