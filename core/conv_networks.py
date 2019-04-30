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
        nn.init.kaiming_uniform_(self.weight_mu)
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1).uniform_(rho_init,rho_init))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-0.2,0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(rho_init,rho_init))
        
        if init_type != 'random':
            nn.init.uniform_(self.weight_rho,0.541,0.541)
            
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
            
    
class BayesianConv2D(_BayesianConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, init_type = 'random', rho_init = -5):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesianConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, init_type = 'random', rho_init = -5)
    
    def forward(self, input, sample = False):
        if sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    
    def variance_init(self):
        
        self.weight_rho.data = torch.Tensor(self.out_features, self.in_features).uniform_(self.rho_init,self.rho_init).cuda()
        self.bias_rho.data = torch.Tensor(self.out_features).uniform_(self.rho_init,self.rho_init).cuda()

class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -5):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.l1 = BayesianConv2D(1, 32, 3, init_type=init_type, rho_init=rho_init)
        self.l2 = BayesianConv2D(32, 64, 3, init_type=init_type, rho_init=rho_init)
        self.l3 = BayesianConv2D(64, 10, 3, init_type=init_type, rho_init=rho_init)
        
#         self.s1 = torch.nn.Linear(28*28, 100)
#         self.s2 = torch.nn.Linear(100, 10)

#         self.layer_arr = [self.l1, self.l2, self.l3, self.s1, self.s2]
        self.layer_arr = [self.l1, self.l2, self.l3]


    def forward(self, x, sample=False):
        s = x
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = self.l3(x, sample)
#         x = nn.AvgPool2d(x, x.size()[2:]) 
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.squeeze(x)
        x = F.log_softmax(x, dim=1)
        
#         s = s.view(-1,28*28)
#         s = F.relu(self.s1(s))
#         s = F.relu(self.s2(s))
#         s = F.log_softmax(s, dim=1)
#         return x, s
        return x
    
    def sample_elbo(self, data, target, BATCH_SIZE, samples=5, saver_net = None):
        # outputs = torch.zeros(samples, BATCH_SIZE, 10).to(DEVICE)
        outputs_x = torch.zeros(samples, BATCH_SIZE, 10).cuda()
#         outputs_s = None
        
        for i in range(samples):
#             outputs_x[i], outputs_s = self(data, sample=True)
            outputs_x[i] = self(data, sample=True)

        loss_x = F.nll_loss(outputs_x.mean(0), target, reduction='sum')
#         loss_s = F.nll_loss(outputs_s, target)
        
#         loss = loss_x + loss_s
        loss = loss_x
        
        return loss
    def var_init(self):
#         self.l1.variance_init()
#         self.l2.variance_init()
        self.l3.variance_init()
        return


