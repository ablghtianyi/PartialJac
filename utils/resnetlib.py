import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
import os


######################################################################
#####Models#####
# Model adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py and https://github.com/zhuchen03/gradinit/blob/master/models/resnet_cifar.py
######################################################################

class ResNet(nn.Module):

    def __init__(self, block, layers, sizes=[32, 16, 8], num_classes=10, use_norm='LN', use_zero_init=False, init_multip=1, mu=1, **kwargs):
        super(ResNet, self).__init__()

        self.num_layers = sum(layers)
        self.inplanes = 16
        self.size = 32
        self.use_norm = use_norm
        self.mu = mu
        if use_norm == 'BN':
            self.input_layer = nn.Sequential(conv3x3(3, 16, bias=not use_norm),
                                             torch.nn.BatchNorm2d(16), 
                                             nn.ReLU(inplace=False))
        elif use_norm == 'LN':
            self.input_layer = nn.Sequential(conv3x3(3, 16, bias=True),
                                              torch.nn.LayerNorm([16, 32, 32]), 
                                              nn.ReLU(inplace=False))
        else:
            self.input_layer = nn.Sequential(conv3x3(3, 16, bias=not use_norm),
                                             nn.ReLU(inplace=False))
        self.layer1 = self._make_layer(block, 16, layers[0], sizes[0])
        self.layer2 = self._make_layer(block, 32, layers[1], sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], sizes[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, num_classes)

        if use_zero_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)
        else:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                    if m.bias is not None:
                        m.bias.data.zero_()
        
        for id, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.weight.data.shape[1] == 3:
                    pass
                else:
                    m.weight.data *= init_multip
                if m.bias is not None:
                    m.bias.data *= init_multip
            elif isinstance(m, torch.nn.Linear):
                m.weight.data *= 0.1
                # if m.bias is not None:
            #         m.bias.data *= init_multip

    def _make_layer(self, block, planes, blocks, size, stride=1):
        downsample = None
        if stride != 1:
            if self.use_norm == 'BN':
                downsample = nn.Sequential(
                    nn.AvgPool2d(1, stride=stride),
                    torch.nn.BatchNorm2d(self.inplanes, track_running_stats=False),
                )
            elif self.use_norm == 'LN':
                downsample = nn.Sequential(
                    nn.AvgPool2d(1, stride=stride),
                    torch.nn.LayerNorm([self.inplanes, size, size]),
                )
            else:
                downsample = nn.Sequential(nn.AvgPool2d(1, stride=stride))
        layers = []
        layers.append(block(
            self.inplanes, planes, size, self.mu, stride, downsample, use_norm=self.use_norm))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, size, self.mu, use_norm=self.use_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
    

def resnet110(**kwargs):
    """Constructs a ResNet-110 model.
    """
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


######################################################################
#####Layers#####
######################################################################

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, size, mu, stride=1, downsample=None, use_norm='LN', bias=True):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.stride = stride
        self.use_norm = use_norm
        self.mu = mu
        bs_size = int(512 / inplanes)

        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        if self.use_norm == 'BN':
            self.norm1 = torch.nn.BatchNorm2d(inplanes)
        elif self.use_norm == 'LN':
            self.norm1 = torch.nn.LayerNorm([inplanes, bs_size, bs_size])
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        if self.use_norm == 'BN':
            self.norm2 = torch.nn.BatchNorm2d(planes)
        elif self.use_norm == 'LN':
            self.norm2 = torch.nn.LayerNorm([planes, size, size])
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = x

        if self.use_norm == 'BN':
            out = self.norm1(out)
        elif self.use_norm == 'LN':
            out = self.norm1(out)
        out = self.relu(out)
        out = self.conv1(out)
        
        if self.use_norm == 'BN':
            out = self.norm2(out)
        elif self.use_norm == 'LN':
            out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)
            
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity * self.mu

        return out


######################################################################
#####Training#####
######################################################################

@torch.no_grad()
def check_accuracy(
    loader: DataLoader, model: nn.Module, dtype, device: str, scaler
) -> tuple:
    
    num_correct = 0
    num_samples = 0
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    x_wrong = []
    for x, y in loader:

        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        if scaler is None:
            scores = model(x)
        else:
            with torch.cuda.amp.autocast():
                scores = model(x)
                
        _, preds = scores.max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        x_wrong.append(x[y != preds])

    acc = float(num_correct) / num_samples

    return num_correct, num_samples, acc


@torch.no_grad()
def test_loss(model: nn.Module, loader_test: DataLoader, 
              losstype: str, dtype, device: str, scaler):
    
    if losstype == 'MSE':
        criterion = nn.MSELoss()
    elif losstype == 'CSE':
        criterion = nn.CrossEntropyLoss()
    
    model.to(device=device)
    loss = 0
    for x,y in loader_test:
        model.eval()  # put model to training mode
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)
        if scaler is None:
            scores = model(x).squeeze()
            if losstype == 'MSE':
                loss += criterion(
                    scores, 
                    F.one_hot(y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
                )
            elif losstype == 'CSE':
                loss += criterion(scores, y)
        else:
            with torch.cuda.amp.autocast():
                scores = model(x).squeeze()
                if losstype == 'MSE':
                    loss += criterion(
                        scores, 
                        F.one_hot(y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
                    )
                elif losstype == 'CSE':
                    loss += criterion(scores, y)
    return loss / len(loader_test)


def train_one_epoch(
    model: nn.Module, optimizer,  time: int, 
    loader_train: DataLoader, loader_val: DataLoader, 
    dtype, device: str, stopwatch:int=0, losstype: str = 'MSE',
    scheduler=None, if_data:bool=True, verbose:bool=True,
    scaler=None,
    ) -> list:
    
    # Make data dic, contains training data
    data = {'tr_acc': [], 'val_acc': [], 'loss': [], 'val_loss': [],
            'jac': [], 'grad': [], 'grad0': [], 'gradf': [],
            'time':[]}

    model.to(device=device)  # move the model parameters to CPU/GPU
    if losstype == 'MSE':
        criterion = nn.MSELoss()
    elif losstype == 'CSE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError('Choose only MSE or CSE!')

    stopwatch = stopwatch

    for t, (x, y) in enumerate(loader_train):
        
        data['time'] = stopwatch
        if stopwatch == time:
            break

        stopwatch += 1
        model.train()  # put model to training mode
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        
        if scaler is None:
            scores = model(x).squeeze()
            if losstype == 'MSE':
                loss = criterion(
                    scores, 
                    F.one_hot(y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
                )
            elif losstype == 'CSE':
                loss = criterion(scores, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                scores = model(x).squeeze()
                if losstype == 'MSE':
                    loss = criterion(
                        scores, 
                        F.one_hot(y, num_classes=scores.shape[-1]).to(device=device, dtype=dtype)
                    )
                elif losstype == 'CSE':
                    loss = criterion(scores, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if if_data:
            data['loss'].append(loss.detach().cpu().numpy())
        
    if scheduler is not None:
        scheduler.step()
    
    if if_data:
        with torch.no_grad():
            num_correct, num_samples, running_train = check_accuracy(
                loader_train, model, dtype, device, scaler)
            data['tr_acc'].append(running_train)
            num_correct, num_samples, running_val = check_accuracy(
                loader_val, model, dtype, device, scaler)
            data['val_acc'].append(running_val)
            data['val_loss'].append(test_loss(
                model, loader_val, losstype, dtype, device, scaler
            ).detach().clone().cpu().item())
    
    if verbose:
        print('TRAIN: {0:.2f},  TEST: {1:.2f}'.format(running_train, running_val))

    return data


######################################################################
#####Tools#####
######################################################################

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)