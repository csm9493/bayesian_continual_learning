import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle


def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 5):
    if tasknum>5:
        tasknum = 5
    data = {}
    taskcla = []
    size = [1, 28, 28]
    
    # Pre-load
    # MNIST
    mean = (0.1307,)
    std = (0.3081,)
    dat = {}
    dat['train'] = datasets.MNIST('../dat/', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    dat['test'] = datasets.MNIST('../dat/', train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    for i in range(5):
        data[i] = {}
        data[i]['name'] = 'split_mnist-{:d}'.format(i)
        data[i]['ncla'] = 2
        data[i]['train'] = {'x': [], 'y': []}
        data[i]['test'] = {'x': [], 'y': []}
    train_task_arr = [0,0,0,0,0]
    test_task_arr = [0,0,0,0,0]
    for s in ['train', 'test']:
        loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        for image, target in loader:
            task_idx = target.numpy()[0] // 2
            if s == 'train':
                train_task_arr[task_idx] = train_task_arr[task_idx] + 1
            if s == 'test':
                test_task_arr[task_idx] = test_task_arr[task_idx] + 1
                
            data[task_idx][s]['x'].append(image)
            data[task_idx][s]['y'].append(target.numpy()[0]%2)
    
    print(train_task_arr)
    print(test_task_arr)
    for i in range(5):
        for s in ['train', 'test']:
            data[i][s]['x'] = torch.stack(data[i][s]['x'])
            data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
    
        
    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    
    return data, taskcla, size
