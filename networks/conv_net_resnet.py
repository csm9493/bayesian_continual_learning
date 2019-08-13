import sys
import torch
import torch.nn as nn
from utils import *

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self, inputsize, taskcla):
        super(Net, self).__init__()
        ncha,size,_=inputsize
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
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
        return layers
    
    def forward(self, x):
        for i in range(self.blocks):
            x = self.layer3[i](x, )
        for i in range(self.blocks):
            x = self.layer4[i](x, )

        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](x))
        
        return y
    