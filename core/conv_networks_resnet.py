import sys
import torch
import torch.nn as nn
from utils import *
from bayes_layer import BayesianConv2D

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, rho_init = -4.6001):
    """3x3 convolution with padding"""
    return BayesianConv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, rho_init = rho_init)

def conv1x1(in_planes, out_planes, stride=1, rho_init = -4.6001):
    """1x1 convolution"""
    return BayesianConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, rho_init = rho_init)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, rho_init = -4.6001, sample = False):
        super(BasicBlock, self).__init__()
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sample = sample
        self.conv1 = conv3x3(inplanes, planes, stride, rho_init = rho_init)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, rho_init = rho_init)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, sample = False):
        identity = x
        out = self.conv1(x, sample)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, sample)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -4.6001):
        super(BayesianConvNetwork, self).__init__()
        ncha,size,_=inputsize
        self.rho_init = -4.6001
        self.taskcla = taskcla
        
        self.inplanes = 128
        self.blocks = 2
        
        self.layer3 = self._make_layer(BasicBlock, 64, self.blocks, stride=2, dilate=False)
        self.layer4 = self._make_layer(BasicBlock, 128, self.blocks, stride=2, dilate=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(128,n))
        self.relu = torch.nn.ReLU()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, rho_init = self.rho_init),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, rho_init = self.rho_init))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rho_init = self.rho_init))

#         return nn.Sequential(*layers)
        return layers
    
    def forward(self, x, sample = False):
        for i in range(self.blocks):
            x = self.layer3[i](x, sample=sample)
        for i in range(self.blocks):
            x = self.layer4[i](x, sample=sample)

        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](x))
        
        return y
    