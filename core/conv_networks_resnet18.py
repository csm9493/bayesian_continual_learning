import sys
import torch
import torch.nn as nn
from utils import *
from bayes_layer import BayesianConv2D

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, ratio = 0.25):
    """3x3 convolution with padding"""
    return BayesianConv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, ratio = ratio)

def conv1x1(in_planes, out_planes, stride=1, ratio = 0.25):
    """1x1 convolution"""
    return BayesianConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, ratio = ratio)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio = 0.25, sample = False):
        super(BasicBlock, self).__init__()
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sample = sample
        self.conv1 = conv3x3(inplanes, planes, stride, ratio = ratio)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, ratio = ratio)
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
            identity = self.downsample[0](x, sample)
            identity = self.downsample[1](identity)

        out += identity
        out = self.relu(out)

        return out


class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, ratio = 0.25):
        super(BayesianConvNetwork, self).__init__()
        ncha,size,_=inputsize
        self.ratio = ratio
        self.taskcla = taskcla
        
        self.inplanes = 64
        self.blocks = 2
        self.conv1 = BayesianConv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, self.blocks)
        self.layer2 = self._make_layer(BasicBlock, 128, self.blocks, stride=2, dilate=False)
        self.layer3 = self._make_layer(BasicBlock, 256, self.blocks, stride=2, dilate=False)
        self.layer4 = self._make_layer(BasicBlock, 512, self.blocks, stride=2, dilate=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(512,n))
        self.relu = torch.nn.ReLU()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = [conv1x1(self.inplanes, planes * block.expansion, stride, ratio = self.ratio),
                      nn.BatchNorm2d(planes * block.expansion),
                     ]
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, ratio = self.ratio))
        self.inplanes = planes * block.expansion
        layers.append(block(self.inplanes, planes, ratio = self.ratio))

#         return nn.Sequential(*layers)
        return layers
    
    def forward(self, x, sample = False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x,sample))))
        
        x = self.layer1[0](x,sample)
        x = self.layer1[1](x,sample)
        x = self.layer2[0](x,sample)
        x = self.layer2[1](x,sample)
        x = self.layer3[0](x,sample)
        x = self.layer3[1](x,sample)
        x = self.layer4[0](x,sample)
        x = self.layer4[1](x,sample)

        x = self.avgpool(x)
        
        x = x.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](x))
        
        return y
    