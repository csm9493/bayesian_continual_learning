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


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, model_old, nepochs=100, sbatch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None, log_name=None):
        self.model = model
        self.model_old = model_old
        self.fisher=None

        file_name = log_name
        self.logger = utils.logger(file_name=file_name, resume=False, path='./result_data/csvdata/', data_format='csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.lamb=args.lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

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

            self.train_epoch(t, xtrain, ytrain)

            clock1 = time.time()
            train_loss, train_acc = self.eval(xtrain, ytrain)
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / xtrain.size(0),
                1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0), train_loss, 100 * train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

            # save log for current task & old tasks at every epoch
            self.logger.add(epoch=(t * self.nepochs) + e, task_num=t + 1, valid_loss=valid_loss, valid_acc=valid_acc)
            for task in range(t):
                xvalid_t = data[task]['valid']['x'].cuda()
                yvalid_t = data[task]['valid']['y'].cuda()
                valid_loss_t, valid_acc_t = self.eval(xvalid_t, yvalid_t)
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

            self.model_old = deepcopy(self.model)
            utils.freeze_model(self.model_old)  # Freeze the weights

            # for n, m in self.model.named_children():
            #     print(n, m.weight.sigma.min())

        # Restore best
        utils.set_model_(self.model, best_model)

        self.logger.save()

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self.criterion)
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option
                #self.fisher[n]=0.5*(self.fisher[n]+fisher_old[n])

        return

    def train_epoch(self,t, x, y):
        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = x[b]
            targets = y[b]

            # Forward current model
            # outputs = self.model.forward(images)
            mini_batch_size = len(targets)
            loss = self.model.sample_elbo(images, targets, mini_batch_size)
            loss = self.custom_regularization(t, self.model_old, self.model, mini_batch_size, loss)

            # for (n,p_old), (_, p) in zip(self.model_old.named_parameters(), self.model.named_parameters()):
            #     print(n, p_old.type(), p.type())
            # print(images.type())
            # print(targets.type())
            # exit()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
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

    def eval(self, x, y, samples=10):
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

        mini_batch_size = len(targets)
        loss = F.nll_loss(output, targets, reduction='sum')/len(targets)
        loss = self.custom_regularization(t, self.model_old, self.model, mini_batch_size, loss)

        return loss


# custom regularization

    def custom_regularization(self,t, saver_net, trainer_net, mini_batch_size, loss=None):
        mean_reg = 0
        sigma_reg = 0

        # net1, net2에서 각 레이어에 있는 mean, sigma를 이용하여 regularization 구현

        # 만약 BayesianNetwork 이면
        if isinstance(saver_net, Net) and isinstance(trainer_net, Net):

            # 각 모델에 module 접근
            for (n, saver_layer), (_, trainer_layer) in zip(saver_net.named_children(), trainer_net.named_children()):
                    # calculate mean regularization
                trainer_mu = trainer_layer.weight_mu
                saver_mu = saver_layer.weight_mu

                trainer_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
                saver_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))

                # mean_reg += lambda_*(torch.div(trainer_layer.weight_mu, saver_layer.weight_rho)-torch.div(trainer_layer.weight_mu, trainer_layer.weight_rho)).norm(2)
                if t ==0:
                    mean_reg += (torch.div(trainer_mu, saver_sigma) - torch.div(saver_mu, saver_sigma)).norm(2)
                else:
                    mean_reg += (torch.div(trainer_mu * self.fisher[n+'.weight_mu'], saver_sigma) - torch.div(saver_mu * self.fisher[n+'.weight_mu'], saver_sigma)).norm(2)

                if args.use_sigmamax:
                    mean_reg = mean_reg * saver_sigma.max()

                # calculate sigma_reg regularization

                # sigma_reg += torch.sum(torch.div(trainer_layer.weight_rho, saver_layer.weight_rho) - torch.log(torch.div(trainer_layer.weight_rho, saver_layer.weight_rho)))
                sigma_reg += torch.sum(torch.div(trainer_sigma **2 , saver_sigma **2) - torch.log(
                    torch.div(trainer_sigma **2, saver_sigma **2)))

            sigma_reg = sigma_reg / (mini_batch_size * 2)
            mean_reg = mean_reg / (mini_batch_size * 2)

            loss = loss / mini_batch_size

        #             print (mean_reg, sigma_reg) # regularization value 확인
        loss = loss + self.lamb * mean_reg + sigma_reg

        return loss