import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
from torch._six import inf
import pandas as pd
from arguments import get_args
args = get_args()

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=torch.autograd.Variable(x[b],volatile=False)
        target=torch.autograd.Variable(y[b],volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs=model.forward(images)
        if args.one_output:
            output = outputs
        else:
            output = outputs[t]
        loss=criterion(t,output,target)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n,_ in model.named_parameters():
            fisher[n]=fisher[n]/x.size(0)
    return fisher

########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################

def clip_relevance_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)

    return total_norm

########################################################################################################################

class logger(object):
    def __init__(self, file_name='pmnist2', resume=False, path='./result_data/csvdata/', data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format


    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)


    def save(self):
        return self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))