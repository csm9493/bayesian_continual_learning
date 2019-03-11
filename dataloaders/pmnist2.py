import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import h5py


########################################################################################################################

def get(seed=0, fixed_order=False, pc_valid=0):
    data = {}
    taskcla = []
    size = [1, 28, 28]

    nperm = 10
    #     nperm = 3
    seeds = np.array(list(range(nperm)), dtype=int)
    if not fixed_order:
        seeds = shuffle(seeds, random_state=seed)

    if not os.path.isdir('./mnist-data/binary_pmnist2/'):
        os.makedirs('./mnist-data/binary_pmnist2')
        # Pre-load
        # MNIST
        mean = (0.1307,)
        std = (0.3081,)

        #         dat = {}
        #         dat['train'] = datasets.MNIST('../dat/', train=True, download=True, transform=transforms.Compose(
        #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        #         dat['test'] = datasets.MNIST('../dat/', train=False, download=True, transform=transforms.Compose(
        #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))

        # normalization, load

        filename = 'Permuted_MNIST_task10.hdf5'
        f = h5py.File('./mnist-data/' + filename, 'r')

        X_train_data = (np.array(f['X_train_data']) / 255. - mean[0]) / std[0]
        X_test_data = (np.array(f['X_test_data']) / 255. - mean[0]) / std[0]
        #         X_train_data = np.array(f['X_train_data'])
        #         X_test_data = np.array(f['X_test_data'])
        Y_train_data = np.array(f['Y_train_data'])
        Y_test_data = np.array(f['Y_test_data'])

        for i, r in enumerate(seeds):
            print(i, end=',')
            sys.stdout.flush()
            data[i] = {}
            data[i]['name'] = 'pmnist-{:d}'.format(i)
            data[i]['ncla'] = 10

            for s in ['train', 'test']:

                #                 loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[i][s] = {'x': [], 'y': []}

                #                 for image, target in loader:

                if s == 'train':

                    aux = X_train_data[i]
                    target = Y_train_data

                    image = torch.FloatTensor(aux).view([60000, 1, 28, 28])

                else:

                    aux = X_test_data[i]
                    target = Y_test_data

                    image = torch.FloatTensor(aux).view([10000, 1, 28, 28])

                #                     aux = image.view(-1).numpy()
                #                     aux = shuffle(aux, random_state=r * 100 + i)

                data[i][s]['x'] = image
                data[i][s]['y'] = target

            # "Unify" and save
            for s in ['train', 'test']:
                data[i][s]['x'] = data[i][s]['x'].view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'], os.path.join(os.path.expanduser('./mnist-data/binary_pmnist2'),
                                                         'data' + str(r) + s + 'x.bin'))
                torch.save(data[i][s]['y'], os.path.join(os.path.expanduser('./mnist-data/binary_pmnist2'),
                                                         'data' + str(r) + s + 'y.bin'))
        print()

    else:

        # Load binary files
        for i, r in enumerate(seeds):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 10
            data[i]['name'] = 'pmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(
                    os.path.join(os.path.expanduser('./mnist-data/binary_pmnist2'), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(
                    os.path.join(os.path.expanduser('./mnist-data/binary_pmnist2'), 'data' + str(r) + s + 'y.bin'))

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
