import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianLinear

class BayesianNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783, unitN = 400, single_head = True):
        super().__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.l1 = BayesianLinear(28*28, unitN, init_type, rho_init)
        self.l2 = BayesianLinear(unitN, unitN, init_type, rho_init)
        self.l3 = BayesianLinear(unitN, taskcla[0][1], init_type, rho_init)
        self.drop=torch.nn.Dropout(0.5)
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(unitN,n))
        
        self.layer_arr = [self.l1, self.l2, self.l3]


    def forward(self, x, sample=False):
        x = s = x.view(-1, 28*28)
        x = self.drop(F.relu(self.l1(x, sample)))
        x = self.drop(F.relu(self.l2(x, sample)))
        if self.taskcla[0][1] == 10:
            x = self.l3(x, sample)
            y = F.log_softmax(x, dim=1)
        else:
            y = []
            for t,i in self.taskcla:
                y.append(self.last[t](x))
        
        return y

    
    def sample_elbo(self, data, target, BATCH_SIZE, samples=5):
        outputs = torch.zeros(samples, BATCH_SIZE, 10).cuda()
        
        for i in range(samples):
            outputs[i] = self(data, sample=True)

        loss = F.nll_loss(outputs.mean(0), target, reduction='sum')
        return loss

