import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle


########################################################################################################################

def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 10):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    
    if not os.path.isdir('../dat/binary_pmnist/'):
        os.makedirs('../dat/binary_pmnist')
    # Pre-load
    # MNIST
    mean = torch.Tensor([0.1307])
    std = torch.Tensor([0.3081])
    dat = {}
    dat['train'] = datasets.MNIST('../dat/', train=True, download=True)
    dat['test'] = datasets.MNIST('../dat/', train=False, download=True)
    
    for i in range(tasknum):
        print(i, end=',')
        sys.stdout.flush()
        data[i] = {}
        data[i]['name'] = 'pmnist-{:d}'.format(i)
        data[i]['ncla'] = 10
        permutation = np.random.permutation(28*28)
        for s in ['train', 'test']:
            if s == 'train':
                arr = dat[s].train_data.view(dat[s].train_data.shape[0],-1).float()
                label = torch.LongTensor(dat[s].train_labels)
            else:
                arr = dat[s].test_data.view(dat[s].test_data.shape[0],-1).float()
                label = torch.LongTensor(dat[s].test_labels)
                
            arr = (arr/255 - mean) / std
            data[i][s]={}
            data[i][s]['x'] = arr[:,permutation].view(-1, size[0], size[1], size[2])
            data[i][s]['y'] = label

    # Validation
    for t in data.keys():
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size

########################################################################################################################