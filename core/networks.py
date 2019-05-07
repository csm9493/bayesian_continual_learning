import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu.cuda()
        self.rho = rho.cuda()
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.mu.size()).cuda()
        return self.mu + self.sigma * epsilon   

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_type = 'random', rho_init = -2.783):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rho_init = rho_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_mu)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        
        self.weight_rho = nn.Parameter(torch.Tensor(out_features,1).uniform_(rho_init,rho_init))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(rho_init,rho_init))
        
        if init_type != 'random':
            nn.init.uniform_(self.weight_rho,0.541,0.541)

        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        self.normal = torch.distributions.Normal(0,1)

    def forward(self, input, sample=False):
        if sample:
            epsilon = self.normal.sample(self.weight_mu.size()).cuda()
            sigma = torch.log1p(torch.exp(self.weight_rho))
            mu = self.weight_mu
            weight = mu + sigma * epsilon
            
            epsilon = self.normal.sample(self.bias_mu.size()).cuda()
            sigma = torch.log1p(torch.exp(self.bias_rho))
            mu = self.bias_mu
            bias = mu + sigma * epsilon
            
        else:
            
            weight = self.weight.mu
            bias = self.bias.mu

        return F.linear(input, weight, bias)

    def variance_init(self):
        
        self.weight_rho.data = torch.Tensor(self.out_features, self.in_features).uniform_(self.rho_init,self.rho_init).cuda()
        self.bias_rho.data = torch.Tensor(self.out_features).uniform_(self.rho_init,self.rho_init).cuda()

class BayesianNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783, unitN = 400, single_head = True):
        super().__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.l1 = BayesianLinear(28*28, unitN, init_type, rho_init)
        self.l2 = BayesianLinear(unitN, unitN, init_type, rho_init)
        self.l3 = BayesianLinear(unitN, taskcla[0][1], init_type, rho_init)
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(400,n))
        
        self.s1 = torch.nn.Linear(28*28, 100)
        self.s2 = torch.nn.Linear(100, 10)

        self.layer_arr = [self.l1, self.l2, self.l3]


    def forward(self, x, sample=False):
        x = s = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        if self.taskcla[0][1] == 10:
            x = self.l3(x, sample)
            y = F.log_softmax(x, dim=1)
        else:
            y = []
            for t,i in self.taskcla:
                y.append(self.last[t](x))
        
        return y

    
    def sample_elbo(self, data, target, BATCH_SIZE, samples=5):
        outputs_x = torch.zeros(samples, BATCH_SIZE, 10).cuda()
        
        for i in range(samples):
            outputs_x[i] = self(data, sample=True)

        loss_x = F.nll_loss(outputs_x.mean(0), target, reduction='sum')
        
        loss = loss_x
        
        return loss
    def var_init(self):
#         self.l1.variance_init()
#         self.l2.variance_init()
        self.l3.variance_init()
        return

