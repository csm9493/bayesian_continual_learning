import sys
import torch
import torch.nn.functional as F
# import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        # self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(ncha*size*size,400)
        self.fc2=torch.nn.Linear(400,400)

#         self.fc3=torch.nn.Linear(800,800)
#         self.last=torch.nn.ModuleList()
#         for t,n in self.taskcla:
#             self.last.append(torch.nn.Linear(400,n))
        self.last=torch.nn.Linear(400,10)


    def forward(self,x):
        h=x.view(x.size(0),-1)
        h=F.relu(self.fc1(h))
        h=self.relu(self.fc2(h))
        h=self.drop(self.relu(self.fc3(h)))
#         y=[]
#         for t,i in self.taskcla:
#             y.append(self.last[t](h))
        y=self.last(h)
        return y
