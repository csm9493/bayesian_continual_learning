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

class AttentionLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        
        #weight init
        self.weight = nn.Parameter(torch.Tensor(in_features).normal_(0,1))
        self.affine_1 = nn.Parameter(torch.Tensor(1).normal_(0,1))
        self.affine_2 = nn.Parameter(torch.Tensor(1).normal_(0,1))
        
    def forward(self, saver_std, trainer_std, attention, s):
        weight = self.weight * attention
        affine_1, affine_2 = self.affine_1, self.affine_2
        a1 = torch.matmul(trainer_std, weight) * affine_1
        a2 = torch.matmul(saver_std, weight) * affine_2
        mask = torch.sigmoid(s*(a1+a2))
        return mask
    
    
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, init_type = 'random', rho_init = -5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if init_type == 'random':
            
            min_value_mu = -0.2
            max_value_mu = +0.2
            
        else:
        
            min_value_mu = 0
            max_value_mu = 0
            
            
        # Weight parameters
        #self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight_mu)
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(rho_init,rho_init))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        
        # Bias parameters        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(rho_init,rho_init))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)


    def forward(self, input, sample=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        return F.linear(input, weight, bias)

    def variance_init(self):
        
        min_value_rho = -2.783
        max_value_rho = -2.783
        
        self.weight_rho.data = torch.Tensor(self.out_features, self.in_features).uniform_(min_value_rho,max_value_rho).cuda() # sigma >= 0
        self.bias_rho.data = torch.Tensor(self.out_features).uniform_(min_value_rho,max_value_rho).cuda()

class BayesianNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, init_type = 'random', rho_init = -5, dropout = False):
        super().__init__()

        ncha,size,_=inputsize
        self.taskcla=taskcla
        self.l1 = BayesianLinear(28*28, 400, init_type, rho_init)
        #self.a1 = AttentionLinear(28*28)
        self.l2 = BayesianLinear(400, 400, init_type, rho_init)
        #self.a2 = AttentionLinear(400)
        self.l3 = BayesianLinear(400, 10, init_type, rho_init)
        #self.a3 = AttentionLinear(10)
        self.layer_arr = [self.l1, self.l2, self.l3]
        #self.layer_arr = [self.l1, self.l2, self.l3, self.a1, self.a2, self.a3]
#         self.layer_arr = [self.l1, self.l2, ]


    def forward(self, x, sample=False, saver_net = None, attention = False, s = 1):
        # def forward(self, input, saver_std, trainer_std, attention, s)
        x = x.view(-1, 28*28)

        if attention:
            saver_std = torch.log1p(torch.exp(saver_net.l1.weight_rho))
            trainer_std = torch.log1p(torch.exp(self.l1.weight_rho))
            mask = self.a1(x, saver_std, trainer_std, torch.ones(28*28), s)
            self.mask1 = mask
            x = x*mask

        x = F.relu(self.l1(x, sample))
        
        if attention:
            saver_std = torch.log1p(torch.exp(saver_net.l2.weight_rho))
            trainer_std = torch.log1p(torch.exp(self.l2.weight_rho))
            mask = self.a2(x, saver_std, trainer_std, mask, s)
            self.mask2 = mask
            x = x*mask
        
        x = F.relu(self.l2(x, sample))
            
        if attention:
            saver_std = torch.log1p(torch.exp(saver_net.l3.weight_rho))
            trainer_std = torch.log1p(torch.exp(self.l3.weight_rho))
            mask = self.a3(x, saver_std, trainer_std, mask, s)
            self.mask3 = mask
            x = x*mask
        x = self.l3(x, sample)
        x = F.log_softmax(x, dim=1)
        return x
    
    def variance_init(self):
        
        self.l1.variance_init()
        self.l2.variance_init()
        self.l3.variance_init()
    
    def sample_elbo(self, data, target, BATCH_SIZE, samples=5, saver_net = None, attention = False, s = 1):
        # outputs = torch.zeros(samples, BATCH_SIZE, 10).to(DEVICE)
        outputs = torch.zeros(samples, BATCH_SIZE, 10).cuda()

        for i in range(samples):
            outputs[i] = self(data, sample=True, saver_net = saver_net, attention = attention, s = s)
        # print(outputs.type())

        loss = F.nll_loss(outputs.mean(0), target, reduction='sum')
        # loss = F.cross_entropy(outputs.mean(0), target, reduction='sum')

        return loss


"""
trainer_std_arr = []
saver_std_arr = []

for (_, saver_layer), (_, trainer_layer) in zip(self.model_old.named_children(), self.model.named_children()):
    trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
    saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))

    trainer_weight_sigma = trainer_weight_sigma.data.cpu().numpy()
    saver_weight_sigma = saver_weight_sigma.data.cpu().numpy()

    trainer_std_arr.append(np.min(trainer_weight_sigma))
    saver_std_arr.append(np.min(saver_weight_sigma))


trainer_sum, saver_sum = 0, 0
for i in range(len(trainer_std_arr)):
    trainer_sum += trainer_std_arr[i]
    saver_sum += saver_std_arr[i]
trainer_mean = trainer_sum / len(trainer_std_arr)
saver_mean = saver_sum / len(saver_std_arr)

#layer by layer로 update도 시도해보면 좋을듯.

if trainer_mean < saver_mean * self.ratio:
    print("SAVED")
    self.model_old = deepcopy(self.model)
    self.saved_point.append(self.iteration)
    self.saved = True
    if self.ratio<0.99:
        self.ratio = self.ratio * (1 + self.gamma)
"""


"""
def Adaptive(self):
    layer_num = len(self.model.layer_arr)
    for i in range(layer_num):

        trainer_layer = self.model.layer_arr[i]
        saver_layer = self.model_old.layer_arr[i]

        trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
        saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))

        mask_1 = (trainer_weight_sigma < saver_weight_sigma * args.ratio).type(torch.FloatTensor).cuda()
        mask_0 = ((trainer_weight_sigma < saver_weight_sigma * args.ratio)^1).type(torch.FloatTensor).cuda()

        new_weight_mu = self.model.layer_arr[i].weight_mu * mask_1 + self.model_old.layer_arr[i].weight_mu * (mask_0)
        new_weight_rho = self.model.layer_arr[i].weight_rho * mask_1 + self.model_old.layer_arr[i].weight_rho * (mask_0)
        new_bias_mu = self.model.layer_arr[i].weight_mu * mask_1 + self.model_old.layer_arr[i].weight_mu * (mask_0)
        new_bias_rho = self.model.layer_arr[i].weight_rho * mask_1 + self.model_old.layer_arr[i].weight_rho * (mask_0)

        self.model_old.layer_arr[i].weight_mu.data = new_weight_mu
        self.model_old.layer_arr[i].weight_rho.data = new_weight_rho
        self.model_old.layer_arr[i].bias_mu.data = new_bias_mu
        self.model_old.layer_arr[i].bias_rho.data = new_bias_rho

    return
"""