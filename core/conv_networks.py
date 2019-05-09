import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear
import utils

class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -2.783):
        super().__init__()
        
        ncha,size,_=inputsize
        self.taskcla = taskcla
        self.conv1 = BayesianConv2D(ncha,32,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(size,3)
        self.conv2 = BayesianConv2D(32,32,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv3 = BayesianConv2D(32,32,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv4 = BayesianConv2D(32,48,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv5 = BayesianConv2D(48,48,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        s = s//2
        self.conv6 = BayesianConv2D(48,80,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv7 = BayesianConv2D(80,80,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv8 = BayesianConv2D(80,80,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv9 = BayesianConv2D(80,80,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv10 = BayesianConv2D(80,80,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        s = s//2
        self.conv11 = BayesianConv2D(80,128,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv12 = BayesianConv2D(128,128,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv13 = BayesianConv2D(128,128,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv14 = BayesianConv2D(128,128,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        self.conv15 = BayesianConv2D(128,128,kernel_size=3, padding = 'same', init_type=init_type, rho_init=rho_init)
        s = utils.compute_conv_output_size(s,3)
        s = s//8
        
        self.maxpool2 = torch.nn.MaxPool2d(2)
        self.maxpool8 = torch.nn.MaxPool2d(8)
        
        self.relu = torch.nn.ReLU()
        self.fc1 = BayesianLinear(s*s*128,512,init_type = init_type, rho_init = rho_init)
        self.last = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(512,n))

    def forward(self, x, sample=False):
        h = self.relu(self.conv1(x, sample))
        h = self.relu(self.conv2(h, sample))
        h = self.relu(self.conv3(h, sample))
        h = self.relu(self.conv4(h, sample))
        h = self.maxpool2(self.relu(self.conv5(h, sample)))
        h = self.relu(self.conv6(x, sample))
        h = self.relu(self.conv7(h, sample))
        h = self.relu(self.conv8(h, sample))
        h = self.relu(self.conv9(h, sample))
        h = self.maxpool2(self.relu(self.conv10(h, sample)))
        h = self.relu(self.conv11(x, sample))
        h = self.relu(self.conv12(h, sample))
        h = self.relu(self.conv13(h, sample))
        h = self.relu(self.conv14(h, sample))
        h = self.maxpool8(self.relu(self.conv15(h, sample)))
        h = h.view(x.size(0),-1)
        h = self.relu(self.fc1(h, sample))
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


