import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian(object):
    def __init__(self, mu, rho, DEVICE):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
        self.DEVICE = DEVICE
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.DEVICE)
        return self.mu + self.sigma * epsilon
    
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_type = 'random',DEVICE=None ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.DEVICE = DEVICE
        
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
        self.weight = Gaussian(self.weight_mu, self.weight_rho, self.DEVICE)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(min_value_mu,max_value_mu))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(min_value_rho,max_value_rho))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, self.DEVICE)


    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        return F.linear(input, weight, bias)

    def variance_init(self):
        
        min_value_rho = +1
        max_value_rho = +1
        
        self.weight_rho.data = torch.Tensor(self.out_features, self.in_features).uniform_(min_value_rho,max_value_rho) # sigma >= 0
        self.bias_rho.data = torch.Tensor(self.out_features).uniform_(min_value_rho,max_value_rho)

class BayesianNetwork(nn.Module):
    def __init__(self, init_type = 'random', DEVICE = None):
        super().__init__()
        self.l1 = BayesianLinear(28*28, 400, init_type, DEVICE)
        self.l2 = BayesianLinear(400, 10, init_type, DEVICE)
        
        self.layer_arr = [self.l1, self.l2,]
    
    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = self.l2(x, sample)
        x = F.log_softmax(x, dim=1)
        return x
    
    def variance_init(self):
        
        self.l1.variance_init()
        self.l2.variance_init()

    
    def sample_elbo(self, data, target, BATCH_SIZE, DEVICE, samples=2):
        outputs = torch.zeros(samples, BATCH_SIZE, 10).to(DEVICE)

        for i in range(samples):
            outputs[i] = self(data, sample=True)

        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = negative_log_likelihood

        return loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    