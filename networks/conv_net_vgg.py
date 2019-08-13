import sys
import torch
import torch.nn as nn
from utils import *


class Net(nn.Module):
    def __init__(self, inputsize, taskcla):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        
        self.conv1 = nn.Conv2d(ncha,32,kernel_size=3)
        s = compute_conv_output_size(size,3) #84
        self.conv2 = nn.Conv2d(32,32,kernel_size=3)
        s = compute_conv_output_size(s,3) #82
        s = s//2 #40
        self.conv3 = nn.Conv2d(32,64,kernel_size=3)
        s = compute_conv_output_size(s,3) #38
        self.conv4 = nn.Conv2d(64,64,kernel_size=3)
        s = compute_conv_output_size(s,3,) #36
        s = s//2 #18
        self.conv5 = nn.Conv2d(64,128,kernel_size=3)
        s = compute_conv_output_size(s,3) #16
        self.conv6 = nn.Conv2d(128,128,kernel_size=3)
        s = compute_conv_output_size(s,3) #14
        self.conv7 = nn.Conv2d(128,128,kernel_size=3)
        s = compute_conv_output_size(s,3) #12
        s = s//2 #6
        self.conv8 = nn.Conv2d(128,256,kernel_size=3)
        s = compute_conv_output_size(s,3) #4
        self.conv9 = nn.Conv2d(256,256,kernel_size=3)
        s = compute_conv_output_size(s,3) #2
        self.fc1 = nn.Linear(s*s*256,256) #2*2*128
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        h=self.relu(self.conv1(x))
        h=self.relu(self.conv2(h))
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.conv3(h))
        h=self.relu(self.conv4(h))
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.conv5(h))
        h=self.relu(self.conv6(h))
        h=self.relu(self.conv7(h))
        h=self.drop1(self.MaxPool(h))
        
        h=self.relu(self.conv8(h))
        h=self.relu(self.conv9(h))
        h=h.view(x.shape[0],-1)
        h = self.drop2(self.relu(self.fc1(h)))
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        
        return y