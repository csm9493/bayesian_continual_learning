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

    label_map = {}
    train_folders = sorted(os.listdir(root+'/train'))
    for i,folder in enumerate(train_folders):
        folder_path = os.path.join(root+'/train', folder)
        img_list = sorted(os.listdir(folder_path))
        for j,ims in enumerate(img_list):
            
    
    val_images = sorted(os.listdir(root+'/val/images'))
    with open(root+'/val/val_labels.txt') as f:
        val_labels = f.read().splitlines()
    
    img_arr = np.zeros(())
    for i,folder in enumerate(train_folders):
        folder_path = os.path.join(root+'/train', folder+'/images')
        img_list = sorted(os.listdir(folder_path))[500:]
        for j,ims in enumerate(img_list):
            img_path = os.path.join(folder_path, ims)
            img = imread(img_path)
            
    
    mean = np.array([[[122.65, 114.30, 101.18]]])
    std = np.array([[[4.63, 6.31, 8.00]]])
    
    for i,folder in enumerate(train_folders):
        folder_path = os.path.join(root+'/train', folder+'/images')
        label_map[folder] = i
        img_list = sorted(os.listdir(folder_path))[500:]
        for ims in img_list:
            img_path = os.path.join(folder_path, ims)
            img = (imread(img_path) - mean) / std
            
            try:
                img = (img - mean) / std
            except:
                continue
            
            img_tensor = Tensor(img).float()
            task_idx = i//20
            label = i % 20
            data[task_idx]['train']['x'].append(img_tensor)
            data[task_idx]['train']['y'].append(label) 

    for i,ims in enumerate(val_images):
        img_path = os.path.join(root+'/val/images', ims)
        img = imread(img_path)
        
        try:
            img = (img - mean) / std
        except:
            continue
        img_tensor = Tensor(img).float()
        task_idx = label_map[val_labels[i]] // 20
        label = label_map[val_labels[i]] % 20
        data[task_idx]['test']['x'].append(img_tensor)
        data[task_idx]['test']['y'].append(label) 
    
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
                data[i][s]['x'] = torch.stack(data[i][s]['x'])
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

get(seed=0, fixed_order=False, pc_valid=0, tasknum = 10)

