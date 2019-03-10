import sys
import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.conv1=nn.Conv2d(ncha, 32,3)
        self.conv2=nn.Conv2d(32, 32, 3)
        self.conv3=nn.Conv2d(32, 32, 3)
        self.conv4=nn.Conv2d(32, 64, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
#         self.fc3=torch.nn.Linear(800,800)
        self.last=nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(nn.Linear(10 * 10 * 64,n))
            self.last[t].last_layer = True
        return

    def forward(self,x):
        h=self.relu(self.conv1(x))
        h=self.relu(self.conv2(h))
        h=self.relu(self.conv3(h))
        h=self.relu(self.conv4(h))
        h=self.pool(h)
        h=h.view(h.size(0),-1)
        y=[]
        
        for t,i in self.taskcla:
            y.append(self.last[t](h))

        return y
