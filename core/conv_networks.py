import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear

class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        self.conv1 = BayesianConv2D(ncha,64,kernel_size=size//8, init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(size,size//8)
        s = s//2
        self.conv2 = BayesianConv2D(64,128,kernel_size=size//10, init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,size//10)
        s = s//2
        self.conv3 = BayesianConv2D(128,256,kernel_size=2, init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,2)
        s = s//2
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = BayesianLinear(256*s*s,2048, init_type = init_type, rho_init = rho_init)
        self.fc2 = BayesianLinear(2048,2048, init_type = init_type, rho_init = rho_init)
        self.last = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))
            
        self.layer_arr = [self.l1, self.l2, self.l3]


    def forward(self, x, sample=False):
        h = self.maxpool(self.drop1(self.relu(self.conv1(x, sample))))
        h = self.maxpool(self.drop1(self.relu(self.conv2(h, sample))))
        h = self.maxpool(self.drop2(self.relu(self.conv3(h, sample))))
        h = h.view(x.size(0),-1)
        h = self.drop2(self.relu(self.fc1(h, sample)))
        h = self.drop2(self.relu(self.fc2(h, sample)))
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y
    
    def sample_elbo(self, data, target, BATCH_SIZE, samples=5, saver_net = None):
        outputs = torch.zeros(samples, BATCH_SIZE, 10).cuda()
        for i in range(samples):
            outputs[i] = self(data, sample=True)

        loss = F.nll_loss(outputs.mean(0), target, reduction='sum')
        
        return loss
    def var_init(self):
#         self.l1.variance_init()
#         self.l2.variance_init()
        self.l3.variance_init()
        return


