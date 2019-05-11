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
        
        self.conv1 = BayesianConv2D(ncha,64,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        self.conv1_bn = nn.BatchNorm2d(64)
        s = compute_conv_output_size(size,3,padding=1)
        self.conv2 = BayesianConv2D(64,128,kernel_size=3, stride=2, init_type=init_type, rho_init=rho_init)
        self.conv2_bn = nn.BatchNorm2d(128)
        s = compute_conv_output_size(s,3,stride=2)
        self.conv3 = BayesianConv2D(128,256,kernel_size=3, padding=1, init_type=init_type, rho_init=rho_init)
        self.conv3_bn = nn.BatchNorm2d(256)
        s = compute_conv_output_size(s,3,padding=1)
        self.conv4 = BayesianConv2D(256,10,kernel_size=3, stride=2, init_type=init_type, rho_init=rho_init)
        s = compute_conv_output_size(s,3,stride=2)
        self.AvgPool = torch.nn.AvgPool2d(s)
        
        self.relu = torch.nn.ReLU()

    def forward(self, x, sample=False):
        h=self.relu(self.conv1_bn(self.conv1(x,sample)))
        h=self.relu(self.conv2_bn(self.conv2(h,sample)))
        h=self.relu(self.conv3_bn(self.conv3(h,sample)))
        h=self.relu(self.conv4(h,sample))
        y=self.AvgPool(h)
        y=F.log_softmax(y.squeeze(),dim=1)
        
        return y