def main():
    
    ################# setup #################

    import time
    import math

    import torch
    from torch import Tensor
    import torch.nn as nn
    import numpy as np

    import random
    import sys
    import os
    import pickle

    sys.path.append("../utils")

    from utils.resnetlib import resnet110
    from utils.partialjaclib import GradHook

    dtype = torch.float32

    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    print('using device:', device)
    
    init_multips = np.linspace(0.1, 2.0, 20)
    mus = np.linspace(0.0, 1.0, 21)
    
    
    ################# APJN #################
    
    jacs = []
    n_avg = 1

    jacs_avg = []
    for i_avg in range(n_avg):
        
        jacs_mu = []
        for i_mu in range(len(mus)):
            
            jacs_im = []
            for i_im in range(len(init_multips)):
                
                model = resnet110(init_multip=init_multips[i_im], use_norm='LN', mu=mus[i_mu])
                model = model.to(device=device)

                ghooks = []
                # Hook a specific block in resnet
                ghooks.append(GradHook(model.input_layer, 10, device))
                for i in range(len(model.layer1)):
                    ghooks.append(GradHook(model.layer1[i], 10, device))
                for i in range(len(model.layer2)):
                    ghooks.append(GradHook(model.layer2[i], 10, device))
                for i in range(len(model.layer3)):
                    ghooks.append(GradHook(model.layer3[i], 10, device))
                
                
                input = torch.randn((1,3,32,32), device=device, requires_grad=True)
                output = model(input)
                loss = output.mean()
                loss.backward()
                
                temp = []
                for i in range(len(ghooks)):
                    # Return APJN of the block
                    temp.append(ghooks[i].pj.item())
                jacs_im.append(temp)
                
            jacs_mu.append(jacs_im)
            
        jacs_avg.append(jacs_mu)
        
    jacs = np.array(jacs_avg)
    
    data = {
        'model': 'resnet110ln_v2',
        'n_avg': n_avg,
        'mus': mus,
        'init_multips': init_multips,
        'apjns': jacs 
    }
    with open('resnet_results/apjn/resnet110ln_v2.pickle', 'wb') as f:
        pickle.dump(data, f)
        

if __name__ == "__main__":
    main()