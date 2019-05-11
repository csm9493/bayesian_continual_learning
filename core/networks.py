import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianLinear

class BayesianNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783, unitN = 400, split = False, drop = False):
        super().__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.split = split
        self.drop = drop
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.l1 = BayesianLinear(28*28, unitN, init_type, rho_init)
        self.l2 = BayesianLinear(unitN, unitN, init_type, rho_init)
        self.l3 = BayesianLinear(unitN, taskcla[0][1], init_type, rho_init)
        self.last=torch.nn.ModuleList()
        
        if split:
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN,n))
        
    def forward(self, x, sample=False):
        if self.drop:
            x = self.drop1(x.view(-1, 28*28))
            x = self.drop2(F.relu(self.l1(x, sample)))
            x = self.drop2(F.relu(self.l2(x, sample)))
        else:
            x = x.view(-1, 28*28)
            x = F.relu(self.l1(x, sample))
            x = F.relu(self.l2(x, sample))
        
        if self.split:
            y = []
            for t,i in self.taskcla:
                y.append(self.last[t](x))
            
        else:
            x = self.l3(x, sample)
            y = F.log_softmax(x, dim=1)
        
        return y
