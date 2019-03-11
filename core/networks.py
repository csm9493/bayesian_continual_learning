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
        epsilon = self.normal.sample(self.rho.size()).cuda()
        return self.mu + self.sigma * epsilon
    
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_type = 'random'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if init_type == 'random':
            
            min_value_mu = -5
            max_value_mu = +5
            
            min_value_rho = +1
            max_value_rho = +1
            
        else:
        
            min_value_mu = 0
            max_value_mu = 0
            
            min_value_rho = 1
            max_value_rho = 1
            
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(min_value_mu,max_value_mu))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(min_value_rho,max_value_rho)) # sigma >= 0
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(min_value_mu,max_value_mu))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(min_value_rho,max_value_rho))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)


    def forward(self, input, sample=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
            print('weight is sampled')
        else:
            weight = self.weight.mu
            bias = self.bias.mu
            print('weight is not sampled')

        return F.linear(input, weight, bias)

    def variance_init(self):
        
        min_value_rho = +1
        max_value_rho = +1
        
        self.weight_rho.data = torch.Tensor(self.out_features, self.in_features).uniform_(min_value_rho,max_value_rho).cuda() # sigma >= 0
        self.bias_rho.data = torch.Tensor(self.out_features).uniform_(min_value_rho,max_value_rho).cuda()

class BayesianNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random'):
        super().__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla

        self.l1 = BayesianLinear(28*28, 400, init_type)
        self.l2 = BayesianLinear(400, 10, init_type)
        # self.l3 = BayesianLinear(400, 10, init_type)
        
        # self.layer_arr = [self.l1, self.l2, self.l3,]
        self.layer_arr = [self.l1, self.l2, ]

    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        # x = F.relu(self.l2(x, sample))
        x = self.l2(x, sample)
        # x = F.log_softmax(x, dim=1)
        return x
    
    def variance_init(self):
        
        self.l1.variance_init()
        self.l2.variance_init()
        # self.l3.variance_init()
    
    def sample_elbo(self, data, target, BATCH_SIZE, samples=2):
        # outputs = torch.zeros(samples, BATCH_SIZE, 10).to(DEVICE)
        outputs = torch.zeros(samples, BATCH_SIZE, 10).cuda()

        for i in range(samples):
            outputs[i] = self(data, sample=True)
        # print(outputs.type())

        # loss = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = F.cross_entropy(outputs.mean(0), target, size_average=False, reduction='sum')

        return loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    