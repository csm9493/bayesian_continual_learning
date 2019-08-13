import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
from torch import Tensor

def split_mini_imagenet_loader(root):
    
    data = {}
    for i in range(10):
        data[i] = {}
        data[i]['name'] = 'split_mini_imagenet-{:d}'.format(i)
        data[i]['ncla'] = 10
        data[i]['train'] = {'x': [], 'y': []}
        data[i]['test'] = {'x': [], 'y': []}

    folders = sorted(os.listdir(root+'/train'))
    mean = np.array([[[120.66, 114.51, 102.99]]])
    std = np.array([[[70.12, 68.26, 71.68]]])
    
    for i,folder in enumerate(folders):
        folder_path = os.path.join(root+'/train', folder)
        img_list = sorted(os.listdir(folder_path))
        for idx, ims in enumerate(img_list):
            s = 'train'
            if idx >= 500:
                s = 'test'
            img_path = os.path.join(folder_path, ims)
            img = (imread(img_path) - mean) / std
            img_tensor = Tensor(img).float()
            task_idx = i//10
            label = i % 10
            data[task_idx][s]['x'].append(img_tensor)
            data[task_idx][s]['y'].append(label) 
    
    return data


def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 10):
    
    data = {}
    taskcla = []
    size = [3, 84, 84]
    
    
    # Pre-load
    # mini_imagenet
    if not os.path.isdir('../dat/binary_split_mini_imagenet/'):
        os.makedirs('../dat/binary_split_mini_imagenet')
        root = os.path.dirname(__file__)
        data = split_mini_imagenet_loader(os.path.join(root, '../../dat/Imagenet-mini'))
        
        for i in range(10):
            for s in ['train', 'test']:
                data[i][s]['x']=torch.stack(data[i][s]['x']).view(-1,size[0],size[1],size[2])
                data[i][s]['y']=torch.LongTensor(np.array(data[i][s]['y'],dtype=int)).view(-1)
                torch.save(data[i][s]['x'], os.path.join(os.path.expanduser('../dat/binary_split_mini_imagenet'),
                                                         'data'+str(i)+s+'x.bin'))
                torch.save(data[i][s]['y'], os.path.join(os.path.expanduser('../dat/binary_split_mini_imagenet'),
                                                         'data'+str(i)+s+'y.bin'))
    else:
        data[0] = dict.fromkeys(['name','ncla','train','test'])
        ids=list(shuffle(np.arange(10),random_state=seed))
        print('Task order =',ids)
        for i in range(10):
            data[i] = dict.fromkeys(['name','ncla','train','test'])
            for s in ['train','test']:
                data[i][s]={'x':[],'y':[]}
                data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('../dat/binary_split_mini_imagenet'),
                                                        'data'+str(ids[i])+s+'x.bin'))
                data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('../dat/binary_split_mini_imagenet'),
                                                        'data'+str(ids[i])+s+'y.bin'))
            data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
            data[i]['name']='mini_imagenet-'+str(ids[i-1])
        
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
    
    return data,taskcla, size