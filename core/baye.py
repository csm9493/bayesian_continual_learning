import sys, time, os
import numpy as np
import torch
from copy import deepcopy
import utils
import torch.nn.functional as F
import torch.nn as nn

sys.path.append('..')
from arguments import get_args

args = get_args()

from core.networks import BayesianNetwork as Net
from bayes_layer import BayesianLinear 
from core.conv_networks import BayesianConvNetwork as ConvNet
from bayes_layer import BayesianConv2D 

class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, model_old, nepochs=100, sbatch=256, sample = 5, lr=0.01, lr_min=1e-5, lr_factor=3, lr_patience=5, clipgrad=100, args=None, log_name=None):
   
        self.model = model
        self.model_old = model_old

        file_name = log_name
        self.logger = utils.logger(file_name=file_name, resume=False, path='./result_data/csvdata/', data_format='csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.sample = sample
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.args = args
        self.iteration = 0
        self.epoch = 0
        self.saved_point = []
        self.saved = 0
        self.grad_queue = []
        self.grad_arr = []
        self.saved_iter = 0
        self.grad_sum = 0
        self.split = False
        if args.experiment == 'split_mnist' or args.experiment == 'split_notmnist' or args.experiment == 'split_cifar100':
            self.split = True
        
        # self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.beta = args.beta
        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to', params)
            self.lamb = float(params[0])

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        #return torch.optim.SGD(self.model.parameters(), lr=lr)
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def sample_elbo(self, model, data, target, BATCH_SIZE, samples=5):
        outputs = torch.zeros(samples, BATCH_SIZE, self.nb_classes).cuda()
        
        for i in range(samples):
            if self.split:
#                 outputs[i] = F.log_softmax(model(data, sample=True)[self.tasknum], dim=1)
                outputs[i] = F.log_softmax(model(data, sample=True)[self.tasknum], dim=1)
            else:
                outputs[i] = model(data, sample=True)

        loss = F.nll_loss(outputs.mean(0), target, reduction='sum')
        return loss
    
    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        self.nb_classes = taskcla[t][1]
        self.tasknum = t
        
        # Loop epochs
        for e in range(self.nepochs):
            self.epoch = self.epoch + 1
            # Train
            clock0 = time.time()

            # self.model.variance_init()  # trainer net의 variance크게 init

            # 1. trainer_net training 하는데 regularization을 위해서 saver_net의 정보 이용

            self.train_epoch(xtrain, ytrain)
            

            clock1 = time.time()
            train_loss, train_acc = self.eval(xtrain, ytrain, self.sample, tasknum = t)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / xtrain.size(0),
                1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0), train_loss, 100 * train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(xvalid, yvalid, self.sample, tasknum = t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

            # save log for current task & old tasks at every epoch
            self.logger.add(epoch=(t * self.nepochs) + e, task_num=t + 1, valid_loss=valid_loss, valid_acc=valid_acc)
            for task in range(t):
                xvalid_t = data[task]['valid']['x'].cuda()
                yvalid_t = data[task]['valid']['y'].cuda()
                valid_loss_t, valid_acc_t = self.eval(xvalid_t, yvalid_t, self.sample)
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
#                     lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        if args.conv_net:
                            break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

            utils.freeze_model(self.model_old)  # Freeze the weights
            
#             self.print_log(e)
            
        # Restore best
        utils.set_model_(self.model, best_model)
        self.model_old = deepcopy(self.model)
        self.saved = 1

        self.logger.save()

        return

    def train_epoch(self, x, y):
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

            # Forward current model
            mini_batch_size = len(targets)
            loss = self.sample_elbo(self.model,images, targets, mini_batch_size, self.sample)
            loss = self.custom_regularization(self.model_old, self.model, mini_batch_size, loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

        return

    def eval(self, x, y, samples=5, tasknum = 0):
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

                # Forward
                outputs = torch.zeros(samples, len(targets), self.nb_classes).cuda()

                for i in range(samples):
                    if self.split:
                        outputs[i] = F.log_softmax(self.model(images, sample=args.ensemble)[tasknum],dim=1)
                    else:
                        outputs[i] = self.model(images, sample=args.ensemble)
                loss = F.nll_loss(outputs.mean(0), targets, reduction='sum')
                
                
                _, pred = outputs.mean(0).max(1)
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

    def custom_regularization(self,saver_net, trainer_net, mini_batch_size, loss=None):
        
        sigma_weight_reg_sum = 0
        sigma_bias_reg_sum = 0
        mu_weight_reg_sum = 0
        mu_bias_reg_sum = 0
        L1_mu_weight_reg_sum = 0
        L1_mu_bias_reg_sum = 0
        
        if args.conv_net:
            prev_rho = nn.Parameter(torch.Tensor(3,1,1,1).uniform_(1,1))
            prev_weight_sigma = torch.log1p(torch.exp(prev_rho))
        
        else:
            prev_rho = nn.Parameter(torch.Tensor(28*28,1).uniform_(1,1))
            prev_weight_sigma = torch.log1p(torch.exp(prev_rho))
        
        for (_, saver_layer), (_, trainer_layer) in zip(saver_net.named_children(), trainer_net.named_children()):
            if isinstance(trainer_layer, BayesianLinear)==False and isinstance(trainer_layer, BayesianConv2D)==False:
                continue
            
            # calculate mu regularization
            trainer_weight_mu = trainer_layer.weight_mu
            saver_weight_mu = saver_layer.weight_mu
            trainer_bias_mu = trainer_layer.bias_mu
            saver_bias_mu = saver_layer.bias_mu

            trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
            saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))
            trainer_bias_sigma = torch.log1p(torch.exp(trainer_layer.bias_rho))
            saver_bias_sigma = torch.log1p(torch.exp(saver_layer.bias_rho))
            
            if len(saver_weight_mu.shape) == 4:
                out_features, in_features, _, _ = saver_weight_mu.shape
                curr_sigma = saver_weight_sigma.expand(out_features,in_features,1,1)
                prev_sigma = prev_weight_sigma.permute(1,0,2,3).expand(out_features,in_features,1,1)
            
            else:
                out_features, in_features = saver_weight_mu.shape
                curr_sigma = saver_weight_sigma.expand(out_features,in_features)
                if len(prev_weight_sigma.shape) == 4:
                    feature_size = in_features // (prev_weight_sigma.shape[0])
                    prev_weight_sigma = prev_weight_sigma.reshape(prev_weight_sigma.shape[0],-1)
                    prev_weight_sigma = prev_weight_sigma.expand(prev_weight_sigma.shape[0], feature_size)
                    prev_weight_sigma = prev_weight_sigma.reshape(-1,1)
                prev_sigma = prev_weight_sigma.permute(1,0).expand(out_features,in_features)
            
            L1_sigma = saver_weight_sigma
            L2_sigma = torch.min(curr_sigma, prev_sigma)
            prev_weight_sigma = saver_weight_sigma
            
            mu_weight_reg = (torch.div(trainer_weight_mu-saver_weight_mu, L2_sigma)).norm(2)**2
            mu_bias_reg = (torch.div(trainer_bias_mu-saver_bias_mu, saver_bias_sigma)).norm(2)**2
            
            L1_mu_weight_reg = (torch.div(saver_weight_mu**2,L1_sigma**2)*(trainer_weight_mu - saver_weight_mu)).norm(1)
            L1_mu_bias_reg = (torch.div(saver_bias_mu**2,saver_bias_sigma**2)*(trainer_bias_mu - saver_bias_mu)).norm(1)
            
            std_init = np.log(1+np.exp(self.args.rho))
            
            mu_weight_reg = mu_weight_reg * (std_init ** 2)
            mu_bias_reg = mu_bias_reg * (std_init ** 2)
            L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)
            L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)

            weight_sigma = trainer_weight_sigma**2 / saver_weight_sigma**2
            bias_sigma = trainer_bias_sigma**2 / saver_bias_sigma**2
            
            normal_weight_sigma = trainer_weight_sigma**2
            normal_bias_sigma = trainer_bias_sigma**2
            
            sigma_weight_reg_sum += (weight_sigma - torch.log(weight_sigma)).sum()
            sigma_weight_reg_sum += (normal_weight_sigma - torch.log(normal_weight_sigma)).sum()
            sigma_bias_reg_sum += (bias_sigma - torch.log(bias_sigma)).sum() 
            sigma_bias_reg_sum += (normal_bias_sigma - torch.log(normal_bias_sigma)).sum()
            
            mu_weight_reg_sum += mu_weight_reg
            mu_bias_reg_sum += mu_bias_reg
            L1_mu_weight_reg_sum += L1_mu_weight_reg
            L1_mu_bias_reg_sum += L1_mu_bias_reg
        
        # elbo loss
        loss = loss / mini_batch_size
        # L2 loss
        loss = loss + (mu_weight_reg_sum + mu_bias_reg_sum) / (mini_batch_size*2)
        # L1 loss
        loss = loss + self.saved * (L1_mu_weight_reg_sum + L1_mu_bias_reg_sum) / (mini_batch_size)
        # sigma regularization
        loss = loss + self.beta * (sigma_weight_reg_sum + sigma_bias_reg_sum) / (mini_batch_size*2)
        
        return loss
