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
from core.networks import BayesianLinear as BL
from core.conv_networks import BayesianConvNetwork as ConvNet


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, model_old, nepochs=100, sbatch=64, sample = 5, lr=0.01, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=100, args=None, log_name=None):
   
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
        self.task_boundary = [46901,93801,140701,187601,234501,281401,328301,375201,422101]
        self.saved = 0
        self.grad_queue = []
        self.grad_arr = []
        self.saved_iter = 0
        self.grad_sum = 0

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
        
    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            self.epoch = self.epoch + 1
            # Train
            clock0 = time.time()

            # self.model.variance_init()  # trainer net의 variance크게 init

            # 1. trainer_net training 하는데 regularization을 위해서 saver_net의 정보 이용

            self.train_epoch(xtrain, ytrain)
            

            clock1 = time.time()
            train_loss, train_acc = self.eval(xtrain, ytrain, self.sample)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / xtrain.size(0),
                1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0), train_loss, 100 * train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(xvalid, yvalid, self.sample)
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
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        #break
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
            loss = self.model.sample_elbo(images, targets, mini_batch_size, self.sample, self.model_old)
            loss = self.custom_regularization(self.model_old, self.model, mini_batch_size, loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            

        return

    def eval(self, x, y, samples=5):
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
                outputs_x = torch.zeros(samples, len(targets), 10).cuda()
#                 outputs_s = None

                for i in range(samples):
#                     outputs_x[i], outputs_s = self.model(images, sample=True)
                    outputs_x[i] = self.model(images, sample=True)

                loss_x = F.nll_loss(outputs_x.mean(0), targets, reduction='sum')
#                 loss_s = F.nll_loss(outputs_s, targets)
#                 loss = loss_x+loss_s
                loss = loss_x
                
                
                _, pred = outputs_x.mean(0).max(1)
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
            prev_rho = nn.Parameter(torch.Tensor(1,1,1,1).uniform_(1,1))
            prev_weight_sigma = torch.log1p(torch.exp(prev_rho))
            if isinstance(saver_net, ConvNet) == False or isinstance(trainer_net, ConvNet) == False:
                return
        
        else:
            prev_rho = nn.Parameter(torch.Tensor(28*28,1).uniform_(1,1))
            prev_weight_sigma = torch.log1p(torch.exp(prev_rho))
            if isinstance(saver_net, Net) == False or isinstance(trainer_net, Net) == False:
                return

        for i in range(3):
            trainer_layer = self.model.layer_arr[i]
            saver_layer = self.model_old.layer_arr[i]

            # calculate mu regularization
            trainer_weight_mu = trainer_layer.weight_mu
            saver_weight_mu = saver_layer.weight_mu
            trainer_bias_mu = trainer_layer.bias_mu
            saver_bias_mu = saver_layer.bias_mu

            trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
            saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))
            trainer_bias_sigma = torch.log1p(torch.exp(trainer_layer.bias_rho))
            saver_bias_sigma = torch.log1p(torch.exp(saver_layer.bias_rho))
            
            if args.conv_net:
                out_features, in_features, H, W = saver_weight_mu.shape
                curr_sigma = saver_weight_sigma.expand(out_features,in_features,1,1)
                prev_sigma = prev_weight_sigma.permute(1,0,2,3).expand(out_features,in_features,1,1)
            
            else:
                out_features, in_features = saver_weight_mu.shape
                curr_sigma = saver_weight_sigma.expand(out_features,in_features)
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
    
    
"""
def slow_update(self):
    grad_avg = 0
    grad_cnt = 0
    for (_, layer) in self.model.named_children():
        if isinstance(layer, torch.nn.Linear) == False:
            continue
        grad_cnt += 1
        grad_avg += layer.weight.grad.norm(2).cpu().numpy() * 1000

    grad_avg = grad_avg / grad_cnt 
    grad_sum = sum(self.grad_queue) 

#         if (grad_avg - grad_sum > args.tau) and (self.iteration - self.saved_iter > 30):
    if self.iteration in self.task_boundary:
        grad = grad_avg-grad_sum
        print("SAVED %f"%(args.tau))
        print(grad)
        self.saved_iter = self.iteration
        self.model_old = deepcopy(self.model)
        self.saved_point.append(self.iteration)
        self.grad_arr.append(grad)
        self.saved = 1
        if args.var_init:
            self.model.var_init()

    self.grad_queue.append(grad_avg)
    if len(self.grad_queue) > 30:
        self.grad_queue.pop(0)

    return
    
def print_log(self, e):
    f = open(self.args.output + '_std_value.txt','a')
    min_arr = []
    saver_min_arr = []
    saver_max_arr = []
    min_idx_arr = []
    max_arr = []
    max_idx_arr = []
    saver_mean_arr = []
    mean_arr = []
    var_arr = []
    rho_sum_arr = []
    grad_arr = []

    for (_, saver_layer), (_, trainer_layer) in zip(self.model_old.named_children(), self.model.named_children()):
        if isinstance(saver_layer, torch.nn.Linear) and isinstance(trainer_layer, torch.nn.Linear):
            grad = trainer_layer.weight.grad.norm(2).cpu().numpy()
            grad_arr.append(grad)
            continue

        rho = torch.log1p(torch.exp(trainer_layer.weight_rho))
        saver_rho = torch.log1p(torch.exp(saver_layer.weight_rho))
        rho = rho.data.cpu().numpy()

        saver_rho = saver_rho.data.cpu().numpy()
        saver_min_arr.append(np.min(saver_rho))
        saver_max_arr.append(np.max(saver_rho))
        saver_mean_arr.append(np.mean(saver_rho))

        min_arr.append(np.min(rho))
        min_idx_arr.append(np.argmin(rho))
        max_arr.append(np.max(rho))
        max_idx_arr.append(np.argmax(rho))
        mean_arr.append(np.mean(rho))
        var_arr.append(np.var(rho))
        rho_sum_arr.append(np.sum(rho<np.mean(rho)))

    f.write('Epoch:%d\n'%e)

    f.write(' minimum std:\n')
    for std in min_arr:
        f.write('%f '%(std))

    f.write('\n min idx:\n')
    for idx in min_idx_arr:
        f.write('%d '%(idx))

    f.write('\n maximum std:\n')
    for std in max_arr:
        f.write('%f '%(std))

    f.write('\n max idx:\n')
    for idx in max_idx_arr:
        f.write('%d '%(idx))

    f.write('\n max reg strength:\n')
    for i in range(len(saver_min_arr)):
        lamb = self.lamb
        if self.args.use_sigmamax:
            lamb = lamb * (saver_max_arr[i] ** 2)
        if self.args.use_sigmamean:
            lamb = lamb * (saver_mean_arr[i] ** 2)

        f.write('%f '%(lamb / (saver_min_arr[i] ** 2)))

    f.write('\n mean \n')
    for mean in mean_arr:
        f.write('%f '%(mean))

    f.write('\n var \n')
    for var in var_arr:
        f.write('%f '%(var))

    f.write('\n sum \n')
    for rho_sum in rho_sum_arr:
        f.write('%d '%(rho_sum))

    f.write('\n grad \n')
    for grad in grad_arr:
        f.write('%f '%(grad))

    f.write('saved point per epoch: ')
    for point in self.saved_point:
        f.write('%d '%point)

    f.write('saved grad per epoch: ')
    for grad in self.grad_arr:
        f.write('%f '%grad)

    f.write('\n')
    f.flush()
    



"""