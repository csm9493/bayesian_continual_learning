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
    def __init__(self, inputsize, taskcla, init_type = 'random', FC_ratio = 0.5, CNN_ratio = 0.25):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = BayesianConv2D(ncha,32,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(size,3) #82
        self.conv2 = BayesianConv2D(32,32,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(s,3) #80
        s = s//2 #40
        self.conv3 = BayesianConv2D(32,64,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(s,3) #38
        self.conv4 = BayesianConv2D(64,64,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(s,3) #36
        s = s//2 #18
        self.conv5 = BayesianConv2D(64,128,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(s,3) #16
        self.conv6 = BayesianConv2D(128,128,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(s,3) #14
        self.conv7 = BayesianConv2D(128,128,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(s,3) #12
        s = s//2 #6
        self.conv8 = BayesianConv2D(128,256,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(s,3) #4
        self.conv9 = BayesianConv2D(256,256,kernel_size=3, ratio = CNN_ratio)
        s = compute_conv_output_size(s,3) #2
        self.fc1 = BayesianLinear(s*s*256,256, init_type=init_type, ratio = FC_ratio)
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
        
        h=self.relu(self.conv5(h,sample))
        h=self.relu(self.conv6(h,sample))
        h=self.relu(self.conv7(h,sample))
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.conv8(h,sample))
        h=self.relu(self.conv9(h,sample))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h,sample)))
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y
    
    
# class BayesianConvNetwork(nn.Module):
#     def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -4.6001):
#         super().__init__()
        
#         ncha,size,_=inputsize
#         self.taskcla = taskcla
        
#         self.conv1 = BayesianConv2D(ncha,32,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(size,3) #82
#         self.conv2 = BayesianConv2D(32,32,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(s,3) #80
#         s = s//2 #40
#         self.conv3 = BayesianConv2D(32,64,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(s,3) #38
#         self.conv4 = BayesianConv2D(64,64,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(s,3) #36
#         s = s//2 #18
#         self.conv5 = BayesianConv2D(64,128,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(s,3) #16
#         self.conv6 = BayesianConv2D(128,128,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(s,3) #14
#         self.conv7 = BayesianConv2D(128,128,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(s,3) #12
#         s = s//2 #6
#         self.conv8 = BayesianConv2D(128,256,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(s,3) #4
#         self.conv9 = BayesianConv2D(256,256,kernel_size=3, rho_init = rho_init)
#         s = compute_conv_output_size(s,3) #2
#         self.fc1 = BayesianLinear(s*s*256,256, init_type=init_type, rho_init = rho_init)
#         self.drop1 = nn.Dropout(0.25)
#         self.drop2 = nn.Dropout(0.5)
#         self.MaxPool = torch.nn.MaxPool2d(2)
        
#         self.last=torch.nn.ModuleList()
        
#         for t,n in self.taskcla:
#             self.last.append(torch.nn.Linear(256,n))
#         self.relu = torch.nn.ReLU()

#     def forward(self, x, sample=False):
#         h=self.relu(self.conv1(x,sample))
#         h=self.relu(self.conv2(h,sample))
#         h=self.drop1(self.MaxPool(h))
        
#         h=self.relu(self.conv3(h,sample))
#         h=self.relu(self.conv4(h,sample))
#         h=self.drop1(self.MaxPool(h))
        
#         h=self.relu(self.conv5(h,sample))
#         h=self.relu(self.conv6(h,sample))
#         h=self.relu(self.conv7(h,sample))
#         h=self.drop1(self.MaxPool(h))
        
#         h=self.relu(self.conv8(h,sample))
#         h=self.relu(self.conv9(h,sample))
#         h=h.view(x.shape[0],-1)
#         h = self.drop2(self.relu(self.fc1(h,sample)))
#         y = []
#         for t,i in self.taskcla:
#             y.append(self.last[t](h))
        
#         return y