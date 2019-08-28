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
        self.taskcla = taskcla
        
        self.inplanes = 32
        self.blocks = 2
        
        self.c1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #######################################################
        planes = 32
        stride = 1
        self.c2 = conv3x3(self.inplanes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.c3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample1 =  nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        self.inplanes = planes
        self.c4 = conv3x3(self.inplanes, planes, 1)
        self.bn4 = nn.BatchNorm2d(planes)
        self.c5 = conv3x3(planes, planes)
        self.bn5 = nn.BatchNorm2d(planes)
        #######################################################
        
        #######################################################
        planes = 64
        stride = 2
        self.c6 = conv3x3(self.inplanes, planes, 1)
        self.bn6 = nn.BatchNorm2d(planes)
        self.c7 = conv3x3(planes, planes)
        self.bn7 = nn.BatchNorm2d(planes)
        self.downsample2  = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        self.inplanes = planes
        self.c8 = conv3x3(self.inplanes, planes, 1)
        self.bn8 = nn.BatchNorm2d(planes)
        self.c9 = conv3x3(planes, planes)
        self.bn9 = nn.BatchNorm2d(planes)
        #######################################################
        
        #######################################################
        planes = 128
        stride = 2
        self.c10 = conv3x3(self.inplanes, planes, 1)
        self.bn10 = nn.BatchNorm2d(planes)
        self.c11 = conv3x3(planes, planes)
        self.bn11 = nn.BatchNorm2d(planes)
        self.downsample3 = downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        self.inplanes = planes
        self.c12 = conv3x3(self.inplanes, planes, 1)
        self.bn12 = nn.BatchNorm2d(planes)
        self.c13 = conv3x3(planes, planes)
        self.bn13 = nn.BatchNorm2d(planes)
        #######################################################
        
        #######################################################
        planes = 256
        stride = 2
        self.c14 = conv3x3(self.inplanes, planes, 1)
        self.bn14 = nn.BatchNorm2d(planes)
        self.c15 = conv3x3(planes, planes)
        self.bn15 = nn.BatchNorm2d(planes)
        self.downsample4  = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        self.inplanes = planes
        self.c16 = conv3x3(self.inplanes, planes, 1)
        self.bn16 = nn.BatchNorm2d(planes)
        self.c17 = conv3x3(planes, planes)
        self.bn17 = nn.BatchNorm2d(planes)
        #######################################################
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(512,n))
        self.relu = torch.nn.ReLU()
        
        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),32)
        self.ec2=torch.nn.Embedding(len(self.taskcla),32)
        self.ec3=torch.nn.Embedding(len(self.taskcla),32)
        self.ec4=torch.nn.Embedding(len(self.taskcla),32)
        self.ec5=torch.nn.Embedding(len(self.taskcla),32)
        self.ec6=torch.nn.Embedding(len(self.taskcla),64)
        self.ec7=torch.nn.Embedding(len(self.taskcla),64)
        self.ec8=torch.nn.Embedding(len(self.taskcla),64)
        self.ec9=torch.nn.Embedding(len(self.taskcla),64)
        self.ec10=torch.nn.Embedding(len(self.taskcla),128)
        self.ec11=torch.nn.Embedding(len(self.taskcla),128)
        self.ec12=torch.nn.Embedding(len(self.taskcla),128)
        self.ec13=torch.nn.Embedding(len(self.taskcla),128)
        self.ec14=torch.nn.Embedding(len(self.taskcla),256)
        self.ec15=torch.nn.Embedding(len(self.taskcla),256)
        self.ec16=torch.nn.Embedding(len(self.taskcla),256)
        self.ec17=torch.nn.Embedding(len(self.taskcla),256)
        
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.ec4.weight.data.uniform_(lo,hi)
        self.ec5.weight.data.uniform_(lo,hi)
        self.ec6.weight.data.uniform_(lo,hi)
        self.ec7.weight.data.uniform_(lo,hi)
        self.ec8.weight.data.uniform_(lo,hi)
        self.ec9.weight.data.uniform_(lo,hi)
        self.ec10.weight.data.uniform_(lo,hi)
        self.ec11.weight.data.uniform_(lo,hi)
        self.ec12.weight.data.uniform_(lo,hi)
        self.ec13.weight.data.uniform_(lo,hi)
        self.ec14.weight.data.uniform_(lo,hi)
        self.ec15.weight.data.uniform_(lo,hi)
        self.ec16.weight.data.uniform_(lo,hi)
        self.ec17.weight.data.uniform_(lo,hi)
        #"""

    
    def forward(self, x):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gc4,gc5,gc6,gc7,gc8,gc9,gc10,gc11,gc12,gc13,gc14,gc15,gc16,gc17=masks
        
        x = self.maxpool(self.relu(self.bn1(self.c1(x))))
        x = x*gc1.view(1,-1,1,1).expand_as(x)
        
        #######################################################
        # 32
        identity = self.downsample1(x)
        x = self.relu(self.bn2(self.c2(x)))
        x = x*gc2.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.bn3(self.c3(x))+identity)
        x = x*gc3.view(1,-1,1,1).expand_as(x)
        
        identity = x
        x = self.relu(self.bn4(self.c4(x)))
        x = x*gc4.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.bn5(self.c5(x))+identity)
        x = x*gc5.view(1,-1,1,1).expand_as(x)
        #######################################################
        
        #######################################################
        # 64
        identity = self.downsample2(x)
        x = self.relu(self.bn6(self.c6(x)))
        x = x*gc6.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.bn7(self.c7(x))+identity)
        x = x*gc7.view(1,-1,1,1).expand_as(x)
        
        identity = x
        x = self.relu(self.bn8(self.c8(x)))
        x = x*gc8.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.bn9(self.c9(x))+identity)
        x = x*gc9.view(1,-1,1,1).expand_as(x)
        #######################################################
        
        #######################################################
        # 128
        identity = self.downsample3(x)
        x = self.relu(self.bn10(self.c10(x)))
        x = x*gc10.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.bn11(self.c11(x))+identity)
        x = x*gc11.view(1,-1,1,1).expand_as(x)
        
        identity = x
        x = self.relu(self.bn12(self.c12(x)))
        x = x*gc12.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.bn13(self.c13(x))+identity)
        x = x*gc13.view(1,-1,1,1).expand_as(x)
        #######################################################
        
        #######################################################
        # 256
        identity = self.downsample4(x)
        x = self.relu(self.bn14(self.c14(x)))
        x = x*gc14.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.bn15(self.c15(x))+identity)
        x = x*gc15.view(1,-1,1,1).expand_as(x)
        
        identity = x
        x = self.relu(self.bn16(self.c16(x)))
        x = x*gc16.view(1,-1,1,1).expand_as(x)
        x = self.relu(self.bn17(self.c17(x))+identity)
        x = x*gc17.view(1,-1,1,1).expand_as(x)
        #######################################################

        x = self.avgpool(x)
        
        x = x.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](x))
        
        return y
    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gc4=self.gate(s*self.ec4(t))
        gc5=self.gate(s*self.ec5(t))
        gc6=self.gate(s*self.ec6(t))
        gc7=self.gate(s*self.ec7(t))
        gc8=self.gate(s*self.ec8(t))
        gc9=self.gate(s*self.ec9(t))
        gc10=self.gate(s*self.ec10(t))
        gc11=self.gate(s*self.ec11(t))
        gc12=self.gate(s*self.ec12(t))
        gc13=self.gate(s*self.ec13(t))
        gc14=self.gate(s*self.ec14(t))
        gc15=self.gate(s*self.ec15(t))
        gc16=self.gate(s*self.ec16(t))
        gc17=self.gate(s*self.ec17(t))
        
        return [gc1,gc2,gc3,gc4,gc5,gc6,gc7,gc8,gc9,gc10,gc11,gc12,gc13,gc14,gc15,gc16,gc17]
    