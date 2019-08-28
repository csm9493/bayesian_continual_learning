import sys, time, os
import numpy as np
import random
import torch
from copy import deepcopy
import utils
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
import math

sys.path.append('..')
from arguments import get_args

args = get_args()

from core.networks import BayesianNetwork as Net
from bayes_layer import BayesianLinear 
from core.conv_networks import BayesianConvNetwork as ConvNet
from bayes_layer import BayesianConv2D 
from bayes_layer import _calculate_fan_in_and_fan_out

resnet_model = models.resnet18(pretrained=True).cuda()
feature_extractor = nn.Sequential(*list(resnet_model.children())[:-4])

class Appr(object):
    

    def __init__(self, model, model_old, nepochs=100, sbatch=256, lr=0.001, 
                 lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100, args=None, log_name=None, split=False):

        self.model = model
        self.model_old = model_old
        
        file_name = log_name
        self.logger = utils.logger(file_name=file_name, resume=False, path='./result_data/csvdata/', data_format='csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_rho = args.lr_rho
        self.lr_min = lr / (lr_factor ** 5)
        self.lr_factor = lr_factor
        self.lr_patience = 5
        self.clipgrad = clipgrad
        self.args = args
        self.iteration = 0
        self.epoch = 0
        self.saved = 0
        self.split = split
        self.beta = args.beta
        
        self.param_name = []
        self.prev_std = {}
        self.std_init = {}
        self.lr_scale = {}
        
        for (n, p) in self.model.named_parameters():
            self.param_name.append(n)
            self.lr_scale[n] = 1
            if 'rho' in n:
                self.std_init[n] = self.prev_std[n] =  torch.log1p(torch.exp(p.data))
        
#         for (n, layer) in self.model.named_children():
#             if isinstance(layer, BayesianLinear)==False and isinstance(layer, BayesianConv2D)==False:
#                 continue
            
#             self.std_init[n] = self.prev_std[n] = torch.log1p(torch.exp(layer.weight_rho.data))
            
        
        for (name, p) in self.model.named_parameters():
            print(name)
        
        self.optimizer = self._get_optimizer()
#         Adam(self.model.parameters(), lr=lr, lr_rho=lr_rho, param_name = self.param_name)
        
        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to', params)
            self.lamb = float(para ,ms[0])

        return

    def _get_optimizer(self, lr=None, lr_rho = None):
        if lr is None: lr = self.lr
        if lr_rho is None: lr_rho = self.lr_rho
        return Adam(self.model.parameters(), lr=lr, lr_rho=lr_rho, param_name = self.param_name, lr_scale = self.lr_scale)
#         return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        lr_rho = self.lr_rho
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr, lr_rho)
        
        # Loop epochs
        for e in range(self.nepochs):
            self.epoch = self.epoch + 1
            # Train
            clock0 = time.time()

            # self.model.variance_init()  # trainer net의 variance크게 init

            # 1. trainer_net training 하는데 regularization을 위해서 saver_net의 정보 이용
            
            # CUB 200 xtrain_croped = crop(x_train)
            xtrain_ = xtrain
            xvalid_ = xvalid
            if args.experiment == 'split_CUB200':
                xtrain_ = crop(xtrain, 224, mode='train')
                xvalid_ = crop(xvalid, 224, mode='valid')
                num_batch = len(xtrain)
            else:
                num_batch = xtrain.size(0)
            
            self.train_epoch(t, xtrain_, ytrain)
            
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, xtrain_, ytrain)
            
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / num_batch,
                1000 * self.sbatch * (clock2 - clock1) / num_batch, train_loss, 100 * train_acc), end='')
            # Valid
            
            valid_loss, valid_acc = self.eval(t, xvalid_, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

            # save log for current task & old tasks at every epoch
            self.logger.add(epoch=(t * self.nepochs) + e, task_num=t + 1, valid_loss=valid_loss, valid_acc=valid_acc)
            for task in range(t):
                if args.experiment == 'split_CUB200':
                    xvalid_t=data[task]['valid']['x']
                    xvalid_t = crop(xvalid_t, 224, mode='valid')
                else:
                    xvalid_t=data[task]['valid']['x'].cuda()
                
                yvalid_t=data[task]['valid']['y'].cuda()

                valid_loss_t, valid_acc_t = self.eval(task, xvalid_t, yvalid_t)
                self.logger.add(epoch=(t * self.nepochs) + e, task_num=task + 1, valid_loss=valid_loss_t,
                                valid_acc=valid_acc_t)

            # Adapt lr
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    lr_rho /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        if args.conv_net:
                            pass
#                             break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr, lr_rho)
            print()

            utils.freeze_model(self.model_old)  # Freeze the weights
            
#             self.print_log(e)
            
        # Restore best
        utils.set_model_(self.model, best_model)
        self.model_old = deepcopy(self.model)
        self.saved = 1
        
        # learning rate scale update
        
        for (n, p) in self.model.named_parameters():
            if 'rho' in n:
                self.prev_std[n] = torch.log1p(torch.exp(p.data))
        
        
        self.update_lr_scale()
        self.logger.save()

        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop batches
        
        for i in range(0, len(r), self.sbatch):
            self.iteration += 1
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = x[b]
            targets = y[b]
            if args.experiment == 'split_CUB200':
                images = feature_extractor(images)

            # Forward current model
            mini_batch_size = len(targets)
            
            if self.split:
                output = F.log_softmax(self.model(images, sample=True)[t],dim=1)
            else:
                output = self.model(images, sample=True)
            loss = F.nll_loss(output, targets, reduction='sum')
            loss = self.custom_regularization(loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # manipulate gradients
#             torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
#             if t > 0:
#                 self.restrict_gradient()
            
            self.optimizer.step()

        return

    def eval(self,t,x,y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()
        
        # Loop batches
        with torch.no_grad():
            for i in range(0, len(r), self.sbatch):
                if i + self.sbatch <= len(r):
                    b = r[i:i + self.sbatch]
                else:
                    b = r[i:]
                images = x[b]
                targets = y[b]
                
                if args.experiment == 'split_CUB200':
                    images = feature_extractor(images)

                # Forward
                mini_batch_size = len(targets)
                if self.split:
                    output = F.log_softmax(self.model(images, sample=False)[t],dim=1)
                else:
                    output = self.model(images, sample=False)
                loss = F.nll_loss(output, targets, reduction='sum')
                
                _, pred = output.max(1)
                hits = (pred == targets).float()
                
                total_loss += loss.data.cpu().numpy()
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(b)



        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2

        return self.ce(output, targets) + self.lamb * loss_reg

# custom regularization

    def custom_regularization(self, loss=None):
        
        sigma_weight_reg_sum = 0
        sigma_weight_init_reg_sum = 0
        
        alpha = 1
#         for (n,p) in self.model.named_parameters():
#             if 'rho' in n:
#                 sigma = torch.log1p(torch.exp(p))
#                 weight_sigma = (sigma**2 / self.prev_std[n]**2)
#                 init_weight_sigma = (sigma**2 / self.std_init[n]**2)
                
#                 sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()
#                 sigma_weight_init_reg_sum = sigma_weight_init_reg_sum + (init_weight_sigma - torch.log(init_weight_sigma)).sum()
            
        
        for (n, layer) in self.model.named_children():
            if isinstance(layer, BayesianLinear)==False and isinstance(layer, BayesianConv2D)==False:
                continue
            # calculate sigma regularization
            
            trainer_weight_sigma = torch.log1p(torch.exp(layer.weight_rho))
            saver_weight_sigma = self.prev_std[n + '.weight_rho']
            
            weight_sigma = (trainer_weight_sigma**2 / saver_weight_sigma**2)
            
            init_weight_sigma = (trainer_weight_sigma**2 / self.std_init[n + '.weight_rho']**2)
            
            sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()
            sigma_weight_init_reg_sum = sigma_weight_init_reg_sum + (init_weight_sigma - torch.log(init_weight_sigma)).sum()
            
        # elbo loss
        loss = loss / self.sbatch
        # sigma regularization
#         loss = loss + self.beta * (self.saved * sigma_weight_reg_sum + sigma_weight_init_reg_sum) / (2 * self.sbatch)
        loss = loss + self.beta * (sigma_weight_reg_sum) / (2 * self.sbatch)
            
        return loss
    
    def restrict_gradient(self):
        if args.conv_net:
            in_lr_strength = nn.Parameter(torch.Tensor(3,1,1,1).uniform(1))
        else:
            in_lr_strength = nn.Parameter(torch.Tensor(28*28,1).uniform_(1,1))
        
        for (n,layer) in self.model.named_children():
            if isinstance(layer, BayesianLinear)==False and isinstance(layer, BayesianConv2D)==False:
                continue
            
            out_lr_strength = (self.prev_std[n] / self.std_init[n]) ** 2
            
            if len(layer.weight_mu.shape) == 4:
                out_features, in_features, _, _ = layer.weight_mu.shape
                out_strength = out_lr_strength.expand(out_features,in_features,1,1)
                in_strength = in_lr_strength.permute(1,0).expand(out_features,in_features)
            else:
                out_features, in_features = layer.weight_mu.shape
                out_strength = out_lr_strength.expand(out_features,in_features)
                if len(in_lr_strength.shape) == 4:
                    feature_size = in_features // (in_strength.shape[0])
                    in_lr_strength = in_lr_strength.reshape(in_strength.shape[0],-1)
                    in_lr_strength = in_lr_strength.expand(in_strength.shape[0], feature_size)
                    in_lr_strength = in_lr_strength.reshape(-1,1)
                in_strength = in_lr_strength.permute(1,0).expand(out_features,in_features)
            
            strength = torch.min(out_strength, in_strength)
            in_lr_strength = out_lr_strength
            
            layer.weight_mu.grad.data.mul_(strength)
        
    def update_lr_scale(self):
        if args.conv_net:
            in_lr_strength = nn.Parameter(torch.Tensor(3,1,1,1).uniform(1))
        else:
            in_lr_strength = nn.Parameter(torch.Tensor(28*28,1).uniform_(1,1))
        
        for (n, layer) in self.model.named_children():
            if isinstance(layer, BayesianLinear)==False and isinstance(layer, BayesianConv2D)==False:
                continue
            out_lr_strength = (self.prev_std[n + '.weight_rho'] / self.std_init[n + '.weight_rho']) ** 2

            if len(layer.weight_mu.shape) == 4:
                out_features, in_features, _, _ = layer.weight_mu.shape
                out_strength = out_lr_strength.expand(out_features,in_features,1,1)
                in_strength = in_lr_strength.permute(1,0).expand(out_features,in_features)
            else:
                out_features, in_features = layer.weight_mu.shape
                out_strength = out_lr_strength.expand(out_features,in_features)
                if len(in_lr_strength.shape) == 4:
                    feature_size = in_features // (in_strength.shape[0])
                    in_lr_strength = in_lr_strength.reshape(in_strength.shape[0],-1)
                    in_lr_strength = in_lr_strength.expand(in_strength.shape[0], feature_size)
                    in_lr_strength = in_lr_strength.reshape(-1,1)
                in_strength = in_lr_strength.permute(1,0).expand(out_features,in_features)
            
            strength = torch.min(out_strength, in_strength)
            in_lr_strength = out_lr_strength
            
            self.lr_scale[n + '.weight_mu'] = strength
        
#         for n,p in self.model.named_parameters():
#             if 'rho' in n or 'last' in n:
#                 continue
#             if p.requires_grad:
#                 if p.grad is not None:
#                     out_strength = (self.prev_std[n] / self.std_init[n]) ** 2

#                     if len(p.shape) == 4:
#                         out_features, in_features, _, _ = saver_weight_mu.shape
#                         out_strength = out_strength.expand(out_features,in_features,1,1)
#                         in_strength = in_strength.permute(1,0).expand(out_features,in_features)

#                     else:
#                         out_features, in_features = saver_weight_mu.shape
#                         out_strength = out_strength.expand(out_features,in_features)
#                         if len(in_strength.shape) == 4:
#                             feature_size = in_features // (in_strength.shape[0])
#                             in_strength = in_strength.reshape(in_strength.shape[0],-1)
#                             in_strength = in_strength.expand(in_strength.shape[0], feature_size)
#                             in_strength = in_strength.reshape(-1,1)
#                         in_strength = in_strength.permute(1,0).expand(out_features,in_features)

#                     strength = torch.max(out_strength, in_strength)
                    
#                     in_strength = out_strength
                    
#                     self.lr_scale[n] = strength