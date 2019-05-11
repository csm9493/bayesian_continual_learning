import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear
from utils import *

class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783, drop = False):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        self.drop = drop
        
        self.conv1 = BayesianConv2D(ncha,64,kernel_size=size//8, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(size,size//8)
        s = s//2
        self.conv2 = BayesianConv2D(64,128,kernel_size=size//10, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(s,size//10)
        s = s//2
        self.conv3 = BayesianConv2D(128,256,kernel_size=2, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(s,size//10)
        s = s//2
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1 = BayesianLinear(s*s*64,2048,init_type = init_type, rho_init = rho_init)
        self.fc2 = BayesianLinear(2048,2048,init_type = init_type, rho_init = rho_init)
        self.last = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))

    def forward(self, x, sample=False):
        if self.drop:
            h=self.maxpool(self.drop1(self.relu(self.conv1(x, sample))))
            h=self.maxpool(self.drop1(self.relu(self.conv2(h, sample))))
            h=self.maxpool(self.drop2(self.relu(self.conv3(h, sample))))
            h=h.view(x.size(0),-1)
            h=self.drop2(self.relu(self.fc1(h, sample)))
            h=self.drop2(self.relu(self.fc2(h, sample)))
        else:
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