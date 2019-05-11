import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear
from utils import *

class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        self.drop = drop
        
        self.conv1 = BayesianConv2D(ncha,128,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(size,3,padding=1)
        self.conv2 = BayesianConv2D(128,256,kernel_size=3, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(size,3,stride=2)
        self.conv3 = BayesianConv2D(256,512,kernel_size=3, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(size,3)
        self.conv4 = BayesianConv2D(512,1024,kernel_size=3,padding=1,stride=2 init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(size,3,padding=1,stride=2)
        self.conv5 = BayesianConv2D(1024,2048,kernel_size=3, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(size,3)
        self.conv6 = BayesianConv2D(2048,10,kernel_size=3, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(size,3)
        self.avgpool = torch.nn.AvgPool2d(s)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x, sample=False):
        h=self.maxpool(self.relu(self.conv1(x, sample)))
        h=self.maxpool(self.relu(self.conv2(h, sample)))
        h=self.maxpool(self.relu(self.conv3(h, sample)))
        h=h.view(x.size(0),-1)
        h=self.relu(self.fc1(h, sample))
        h=self.relu(self.fc2(h, sample))
        
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y