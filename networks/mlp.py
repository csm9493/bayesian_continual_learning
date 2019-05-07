import sys
import torch
import torch.nn.functional as F
# import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla, unitN = 400):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(ncha*size*size,unitN)
        self.fc2=torch.nn.Linear(unitN,unitN)

#         self.fc3=torch.nn.Linear(800,800)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(unitN,n))
#         self.last=torch.nn.Linear(400,10)
        self.l=torch.nn.Linear(unitN,taskcla[0][1])


    def forward(self,x):
        h=x.view(x.size(0),-1)
        h=F.relu(self.fc1(h))
        h=F.relu(self.fc2(h))
        if self.taskcla[0][1] == 10:
            h = self.l(h)
        else:
            y = []
            for t,i in self.taskcla:
                y.append(self.last[t](h))
        
        return y