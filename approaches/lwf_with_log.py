import sys,time
import numpy as np
import torch
import utils
sys.path.append('..')
from arguments import get_args
args = get_args()

from networks.mlp_mh import Net

class Appr(object):
    """ Class implementing the Learning Without Forgetting approach described in https://arxiv.org/abs/1606.09282 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=100,T=1,args=None, log_name=None):
        self.model=model
        self.model_old=None

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
        self.lamb=args.lamb          # Grid search = [0.1, 0.5, 1, 2, 4, 8, 10]; best was 2
        self.T=T                # Grid search = [0.5,1,2,4]; best was 1

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
            self.T=float(params[1])

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
            clock0=time.time()
            self.train_epoch(t,xtrain,ytrain)
            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            
            #save log for current task & old tasks at every epoch
            self.logger.add(epoch=(t*self.nepochs)+e, task_num=t+1, valid_loss=valid_loss, valid_acc=valid_acc)
            for task in range(t): 
                xvalid_t=data[task]['valid']['x'].cuda()
                yvalid_t=data[task]['valid']['y'].cuda()
                valid_loss_t,valid_acc_t=self.eval(task,xvalid_t,yvalid_t)
                self.logger.add(epoch=(t*self.nepochs)+e, task_num=task+1, valid_loss=valid_loss_t, valid_acc=valid_acc_t)
            
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

        # Restore best
        utils.set_model_(self.model, best_model)

        self.logger.save()
        
        # Update old
        self.model_old = Net(input_size, taskcla).cuda()
        self.model_old.load_state_dict(self.model.state_dict())
        self.model_old.eval()
        
        utils.freeze_model(self.model_old)
        self.logger.save()
        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]

            # Forward old model
            targets_old=None
            if t>0:
                targets_old=self.model_old.forward(images)

            # Forward current model
            outputs=self.model.forward(images)
            loss=self.criterion(t,targets_old,outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]

            # Forward old model
            targets_old=None
            if t>0:
                targets_old=self.model_old.forward(images)

            # Forward current model
            outputs=self.model.forward(images)
            loss=self.criterion(t,targets_old,outputs,targets)
            output=outputs[t]
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
#             total_loss+=loss.data.cpu().numpy()[0]*len(b)
#             total_acc+=hits.sum().data.cpu().numpy()[0]
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,targets_old,outputs,targets):
        # TODO: warm-up of the new layer (paper reports that improves performance, but negligible)

        # Knowledge distillation loss for all previous tasks
        loss_dist=0
        for t_old in range(0,t):
            loss_dist+=utils.cross_entropy(outputs[t_old],targets_old[t_old],exp=1/self.T)

        # Cross entropy loss
        loss_ce=self.ce(outputs[t],targets)

        # We could add the weight decay regularization mentioned in the paper. However, this might not be fair/comparable to other approaches

        return loss_ce+self.lamb*loss_dist
