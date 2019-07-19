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
    def __init__(self, inputsize, taskcla, init_type = 'random'):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = BayesianConv2D(ncha,128,kernel_size=3, init_type=init_type)
        s = compute_conv_output_size(size,3)
        self.conv2 = BayesianConv2D(128,128,kernel_size=3, init_type=init_type)
        s = compute_conv_output_size(size,3)
        s = s//2
        self.conv3 = BayesianConv2D(128,128,kernel_size=3, init_type=init_type)
        s = compute_conv_output_size(s,3)
        self.conv4 = BayesianConv2D(128,128,kernel_size=3, init_type=init_type)
        s = compute_conv_output_size(s,3)
        s = s//2
        self.fc1 = BayesianLinear(s*s*128,256, init_type=init_type)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, sample=False):
        h=self.relu(self.conv1(x,sample))
        h=self.relu(self.conv2(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=self.relu(self.conv3(h,sample))
        h=self.relu(self.conv4(h,sample))
        h=self.drop1(self.MaxPool(h))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h,sample)))
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y