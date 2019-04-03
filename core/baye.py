import sys, time, os
import numpy as np
import torch
from copy import deepcopy
import utils
import torch.nn.functional as F

sys.path.append('..')
from arguments import get_args

args = get_args()

from core.networks import BayesianNetwork as Net
from core.networks import AttentionLinear
from core.networks import BayesianLinear as BL


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
        self.s = 1

        # self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.lamb = args.lamb
        if len(args.parameter) >= 1:
            params = args.parameter.split(',')
            print('Setting parameters to', params)
            self.lamb = float(params[0])

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        #return torch.optim.SGD(self.model.parameters(), lr=lr)
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def Bernoulli_freeze(self,i):
        for (_, saver_layer), (_, trainer_layer) in zip(self.model_old.named_children(), self.model.named_children()):
            if isinstance(saver_layer, BL) and isinstance(trainer_layer, BL):
                
                rho_saver = saver_layer.weight_rho
                a, b = 1000, -0.0595
                probs = torch.sigmoid(a * (rho_saver + b))
                Bern = torch.distributions.bernoulli.Bernoulli(probs)
                mask = Bern.sample()
                #trainer_layer.weight_mu.grad *= mask
                trainer_layer.weight_mu.grad *= probs
                """
                if i==self.sbatch:
                    print(trainer_layer.weight_mu.grad)
                """
        
        return
    
    
    def print_log(self, e):
        f = open(self.args.output + '_std_value.txt','a')
        min_arr = []
        min_idx_arr = []
        max_arr = []
        max_idx_arr = []
        mean, var, rho_sum = 0, 0, 0
        for (_, layer) in self.model.named_children():
            rho = torch.log1p(torch.exp(layer.weight_rho))
#             rho = layer.weight_rho
            rho = rho.data.cpu().numpy()
            min_arr.append(np.min(rho))
            min_idx_arr.append(np.argmin(rho))
            max_arr.append(np.max(rho))
            max_idx_arr.append(np.argmin(rho))
            mean = np.mean(rho)
            var = np.var(rho)
            rho_sum += np.sum(rho<0.065)
        f.write('Epoch:%d\n'%e)

        f.write(' minimum std:\n')
        for i in range(len(min_arr)):
            f.write('%f '%(min_arr[i]))

        f.write(' minimum std reg strength:\n')
        for i in range(len(min_arr)):
            lamb = self.lamb
            if args.use_sigmamax:
                lamb = lamb * ()
            f.write('%f '%(min_arr[i]))    
        
        f.write('\n min idx:\n')
        for i in range(len(min_arr)):
            f.write('%d '%(min_idx_arr[i]))

        f.write('\n maximum std:\n')
        for i in range(len(min_arr)):
            f.write('%f '%(max_arr[i]))

        f.write('\n max idx:\n')
        for i in range(len(min_arr)):
            f.write('%d '%(max_idx_arr[i]))

        f.write('\n mean: %f'%(mean))
        f.write('\n var: %f'%(var))
        f.write('\n sum: %d'%(rho_sum))

        f.write('\n')
        f.flush()
    
            
    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):

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
                        break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

            #self.model_old = deepcopy(self.model)
            utils.freeze_model(self.model_old)  # Freeze the weights
            
            #self.print_log(e)
            
            
            
            # for n, m in self.model.named_children():
            #     print(n, m.weight.sigma.min())

        # Restore best
        utils.set_model_(self.model, best_model)

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
            # outputs = self.model.forward(images)
            mini_batch_size = len(targets)
            loss = self.model.sample_elbo(images, targets, mini_batch_size, self.sample, self.model_old, self.args.use_Attention, self.s)
            loss = self.custom_regularization(self.model_old, self.model, mini_batch_size, loss)

            # for (n,p_old), (_, p) in zip(self.model_old.named_parameters(), self.model.named_parameters()):
            #     print(n, p_old.type(), p.type())
            # print(images.type())
            # print(targets.type())
            # exit()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.model_old = deepcopy(self.model)
            if args.use_Bernoulli:
                self.Bernoulli_freeze(i)
            
            if args.use_Attention:
                pass
            
            f = open(self.args.output + '_std_value.txt','a')
            min_arr = []
            for (_, layer) in self.model.named_children():
                if isinstance(layer, BL):
                    rho = torch.log1p(torch.exp(layer.weight_rho))
                    rho = rho.data.cpu().numpy()
                    f.write('%f '%(np.min(rho)))
                    
            f.write('\n')
            f.flush()
            
            self.optimizer.step()
            
            # 2. 1 batch가 끝나면 saver_net에 trainet_net을 복사 (weight = mean, sigma)

        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data, target = data.to(DEVICE), target.to(DEVICE)
        #     if data.shape[0] == mini_batch_size:
                # trainer_net.zero_grad()
                # loss = trainer_net.sample_elbo(data, target, mini_batch_size, DEVICE)
                # loss = custom_regularization(saver_net, trainer_net, mini_batch_size, lambda_, loss)
                # loss.backward()
                # print(trainer_net.l2.weight.rho.grad)
                # print(trainer_net.l.weight.rho.grad)
                # print(trainer_net.l1.weight.rho.grad.norm(2))

                # optimizer.step()

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
                outputs = torch.zeros(samples, len(targets), 10).cuda()

                for i in range(samples):
                    outputs[i] = self.model(images, sample=True)
                # print(outputs.type())

                loss = F.nll_loss(outputs.mean(0), targets, reduction='sum')
                _, pred = outputs.mean(0).max(1)
                hits = (pred == targets).float()

                # Log
                #             total_loss+=loss.data.cpu().numpy()[0]*len(b)
                #             total_acc+=hits.sum().data.cpu().numpy()[0]
                total_loss += loss.data.cpu().numpy()
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(b)

                # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # correct += pred.eq(target.view_as(pred)).sum().item()

        # test_loss /= len(test_loader.dataset)
        #
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))


        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                #                 if name.startswith('last'):
                #                     if not args.no_outputreg:
                #                         loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
                #                 else:
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2

        return self.ce(output, targets) + self.lamb * loss_reg


# custom regularization

    def custom_regularization(self,saver_net, trainer_net, mini_batch_size, loss=None):
        mean_weight_reg = 0
        mean_bias_reg = 0
        sigma_weight_reg = 0
        sigma_bias_reg = 0
        mean_sigma = 0
        min_sigma = 0
        sig_max = 0
        mean_weight_reg_sum = 0
        mean_bias_reg_sum = 0
        # net1, net2에서 각 레이어에 있는 mean, sigma를 이용하여 regularization 구현

        # 만약 BayesianNetwork 이면
        if isinstance(saver_net, Net) and isinstance(trainer_net, Net):

            # 각 모델에 module 접근
            for (_, saver_layer), (_, trainer_layer) in zip(saver_net.named_children(), trainer_net.named_children()):
                if isinstance(saver_layer, AttentionLinear) and isinstance(trainer_layer, AttentionLinear):
                    continue
                
                # calculate mean regularization
                trainer_weight_mu = trainer_layer.weight_mu
                saver_weight_mu = saver_layer.weight_mu

                trainer_bias_mu = trainer_layer.bias_mu
                saver_bias_mu = saver_layer.bias_mu

#                 trainer_weight_sigma = trainer_layer.weight_rho
#                 saver_weight_sigma = saver_layer.weight_rho

#                 trainer_bias_sigma = trainer_layer.bias_rho
#                 saver_bias_sigma = saver_layer.bias_rho

                trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
                saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))

                trainer_bias_sigma = torch.log1p(torch.exp(trainer_layer.bias_rho))
                saver_bias_sigma = torch.log1p(torch.exp(saver_layer.bias_rho))

                mean_weight_reg = (torch.div(trainer_weight_mu, saver_weight_sigma) - torch.div(saver_weight_mu, saver_weight_sigma)).norm(2)**2
                mean_bias_reg = (torch.div(trainer_bias_mu, saver_bias_sigma) - torch.div(saver_bias_mu, saver_bias_sigma)).norm(2)**2

                if args.use_sigmamax:
                    sig_max = 
                    mean_weight_reg = mean_weight_reg * ((saver_weight_sigma.max())**2)
                    mean_bias_reg = mean_bias_reg * ((saver_bias_sigma.max())**2)

                mean_sigma = saver_weight_sigma.mean()
                min_sigma = saver_weight_sigma.min()
                # calculate sigma_reg regularization

                # sigma_reg += torch.sum(torch.div(trainer_layer.weight_rho, saver_layer.weight_rho) - torch.log(torch.div(trainer_layer.weight_rho, saver_layer.weight_rho)))
                
                if args.no_sigma_reg == False:
                    sigma_weight_reg += torch.sum(torch.div(trainer_weight_sigma **2 , saver_weight_sigma **2) - torch.log(
                        torch.div(trainer_weight_sigma **2, saver_weight_sigma **2)))
                    sigma_bias_reg += torch.sum(torch.div(trainer_bias_sigma **2 , saver_bias_sigma **2) - torch.log(
                        torch.div(trainer_bias_sigma **2, saver_bias_sigma **2)))


                mean_weight_reg_sum += mean_weight_reg
                mean_bias_reg_sum += mean_bias_reg

        #             print (mean_reg, sigma_reg) # regularization value 확인
        
        f = open('reg_strength.txt','a')
        f.write('%.10f\n'%(self.lamb / (min_sigma ** 2)))
        f.flush()
        loss = loss / mini_batch_size + (self.lamb * (mean_weight_reg_sum + mean_bias_reg_sum) + (sigma_weight_reg + sigma_bias_reg)) /(mini_batch_size * 2)

        return loss