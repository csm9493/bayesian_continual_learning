import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
from torch import Tensor

"""
Loads the train/test set. 
Every image in the dataset is 224x224 pixels and the labels are numbered from 0-199
Set root to point to the Train/Test folders.
"""
def split_tiny_imagenet_loader(root):
    
    data = {}
    for i in range(10):
        data[i] = {}
        data[i]['name'] = 'split_tiny_imagenet-{:d}'.format(i)
        data[i]['ncla'] = 20
        data[i]['train'] = {'x': [], 'y': []}
        data[i]['test'] = {'x': [], 'y': []}

    train_folders = sorted(os.listdir(root+'/train'))
    valid_folders = sorted(os.listdir(root+'/val'))
    
    folders_arr = [train_folders, valid_folders]
    s = ['train','test']
    dir_arr = ['/train', '/val']
    
    mean = np.array([[[122.65, 114.30, 101.18]]])
    std = np.array([[[70.50, 68.45, 71.82]]])
    
    for idx, folders in enumerate(folders_arr):
        for i, folder in enumerate(folders):
            folder_path = os.path.join(root+dir_arr[idx], folder)
            img_list = sorted(os.listdir(folder_path))
            if(i==200):
                print(folder)
            for j,ims in enumerate(img_list):
                img_path = os.path.join(folder_path, ims)
                img = imread(img_path)
                if len(img.shape) == 2:
                    continue

                img = (img - mean) / std
                img_tensor = Tensor(img).float()
                task_idx = i//20
                label = i % 20
                data[task_idx][s[idx]]['x'].append(img_tensor)
                data[task_idx][s[idx]]['y'].append(label) 
        
    return data


def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 10):
    
    data = {}
    taskcla = []
    size = [3, 64, 64]
    
    
    # Pre-load
    # tiny_imagenet
    if not os.path.isdir('../dat/binary_split_tiny_imagenet/'):
        os.makedirs('../dat/binary_split_tiny_imagenet')
        root = os.path.dirname(__file__)
        data = split_tiny_imagenet_loader(os.path.join(root, '../../dat/Imagenet-tiny'))
        
        for i in range(10):
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1,size[0],size[1],size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('../dat/binary_split_tiny_imagenet'),
                                                        'data' + str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('../dat/binary_split_tiny_imagenet'),
                                                        'data' + str(i) + s + 'y.bin'))
    else:
        # Load binary files
        for i in range(10):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 20
            data[i]['name'] = 'split_tiny_imagenet-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_split_tiny_imagenet'),
                                                          'data' + str(i) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_split_tiny_imagenet'),
                                                          'data' + str(i) + s + 'y.bin'))
        
    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['test']['x'].clone()
        data[t]['valid']['y'] = data[t]['test']['y'].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    
    return data, taskcla, size