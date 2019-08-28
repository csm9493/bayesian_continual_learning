import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class BayesianConvNetwork(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.conv1=BayesianConv2D(ncha,64,kernel_size=size//8)
        s=compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=BayesianConv2D(64,128,kernel_size=size//10)
        s=compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=BayesianConv2D(128,256,kernel_size=2)
        s=compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=BayesianLinear(256*s*s,2048)
        self.fc2=BayesianLinear(2048,2048)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))

        return

    def forward(self,x,sample=False):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x,sample))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h,sample))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h,sample))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h,sample)))
        h=self.drop2(self.relu(self.fc2(h,sample)))
        
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y
