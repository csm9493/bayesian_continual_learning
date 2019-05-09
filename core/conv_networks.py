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
    
    def variance_init(self):
        
        self.weight_rho.data = torch.Tensor(self.out_features, self.in_features).uniform_(self.rho_init,self.rho_init).cuda()
        self.bias_rho.data = torch.Tensor(self.out_features).uniform_(self.rho_init,self.rho_init).cuda()

class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.conv1=BayesianConv2D(ncha,64,kernel_size=size//8, init_type=init_type, rho_init=rho_init)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=BayesianConv2D(64,128,kernel_size=size//10, init_type=init_type, rho_init=rho_init)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=BayesianConv2D(128,256,kernel_size=2, init_type=init_type, rho_init=rho_init)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))
            
        self.layer_arr = [self.l1, self.l2, self.l3]


    def forward(self, x, sample=False):
        s = x
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = self.l3(x, sample)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.squeeze(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    
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


