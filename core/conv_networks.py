import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = BayesianConv2D(ncha,512,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        self.conv1_bn = nn.BatchNorm2d(512)
        s = compute_conv_output_size(size,3,padding=1)
        self.conv2 = BayesianConv2D(512,256,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        self.conv2_bn = nn.BatchNorm2d(256)
        s = compute_conv_output_size(size,3,padding=1)
        s = s//2
        self.conv3 = BayesianConv2D(256,256,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        self.conv3_bn = nn.BatchNorm2d(256)
        s = compute_conv_output_size(s,3,padding=1)
        self.conv4 = BayesianConv2D(256,128,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        self.conv4_bn = nn.BatchNorm2d(128)
        s = compute_conv_output_size(s,3,padding=1)
        s = s//2
        self.conv5 = BayesianConv2D(128,128,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        self.conv5_bn = nn.BatchNorm2d(128)
        s = compute_conv_output_size(s,3,padding=1)
        self.conv6 = BayesianConv2D(128,64,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        self.conv6_bn = nn.BatchNorm2d(64)
        s = compute_conv_output_size(s,3,padding=1)
        
        self.AvgPool = torch.nn.AvgPool2d(s)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(64,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, sample=False):
        h=self.relu(self.conv1_bn(self.conv1(x,sample)))
        h=self.relu(self.conv2_bn(self.conv2(h,sample)))
        h=self.MaxPool(h)
        h=self.relu(self.conv3_bn(self.conv3(h,sample)))
        h=self.relu(self.conv4_bn(self.conv4(h,sample)))
        h=self.MaxPool(h)
        h=self.relu(self.conv5_bn(self.conv5(h,sample)))
        h=self.relu(self.conv6_bn(self.conv6(h,sample)))
        h=self.AvgPool(h)
        h=h.squeeze()
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y