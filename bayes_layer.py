import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

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
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2,0.2))
        
        self.weight_rho = nn.Parameter(torch.Tensor(out_features,1).uniform_(rho_init,rho_init))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(rho_init,rho_init))
        
        if init_type != 'random':
            nn.init.uniform_(self.weight_rho,0.541,0.541)

        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

    def forward(self, input, sample=False):
        if sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
            
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        return F.linear(input, weight, bias)

class _BayesianConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding, dilation, transposed, output_padding, groups, bias, init_type, rho_init):
        super(_BayesianConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        nn.init.xavier_uniform_(self.weight_mu)
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1).uniform_(rho_init,rho_init))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(rho_init,rho_init))
        
        if init_type != 'random':
            nn.init.uniform_(self.weight_rho,0.541,0.541)
            
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        
        
class BayesianConv2D(_BayesianConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, init_type = 'random', rho_init = -2.783):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesianConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, init_type, rho_init)
    
    def forward(self, input, sample = False):
        if sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
