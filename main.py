import sys, os, time
import numpy as np
import torch
import torch.nn as nn

import utils
from arguments import get_args


tstart = time.time()

# Arguments

args = get_args()
args_std = np.log(1+np.exp(args.rho))
if args.approach == 'si_with_log':
    log_name = '{}_{}_{}_{}_c_{}_unitN_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed, 
                                                                    args.c, args.unitN, args.batch_size, args.nepochs)
elif args.approach == 'ewc_with_log':
    log_name = '{}_{}_{}_{}_lamb_{}_unitN_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed,
                                                                       args.lamb, args.unitN, args.batch_size, args.nepochs)
elif args.approach == 'baye':
    log_name = '{}_{}_{}_{}_beta_{:.7f}_unitN_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, 
                                                                           args.approach,args.seed,args.beta, 
                                                                           args.unitN,args.batch_size, args.nepochs)
elif args.approach == 'hat':
    log_name = '{}_{}_{}_{}_alpha_{}_unitN_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach, args.seed,
                                                                       args.alpha, args.unitN, args.batch_size, args.nepochs)

if args.conv_net:
    log_name = log_name + '_conv'

if args.output == '':
    args.output = './result_data/' + log_name + '.txt'

    
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

########################################################################################################################
# Split
split = False
notMNIST = False
split_experiment = ['split_mnist', 'split_notmnist', 'split_cifar100','split_cifar10_100','split_pmnist','split_row_pmnist']
if args.experiment in split_experiment:
    split = True
if args.experiment == 'split_notmnist':
    notMNIST = True
    
# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment == 'mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment == 'pmnist' or args.experiment == 'split_pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment == 'row_pmnist' or args.experiment == 'split_row_pmnist':
    from dataloaders import row_pmnist as dataloader
elif args.experiment == 'split_mnist':
    from dataloaders import split_mnist as dataloader
elif args.experiment == 'split_notmnist':
    from dataloaders import split_notmnist as dataloader
elif args.experiment == 'split_cifar100':
    from dataloaders import split_cifar100 as dataloader
elif args.experiment == 'split_cifar10_100':
    from dataloaders import split_cifar10_100 as dataloader
elif args.experiment == 'mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach == 'random':
    from approaches import random as approach
elif args.approach == 'baye':
    from core import baye as approach
elif args.approach == 'baye_hat':
    from core import baye as approach
elif args.approach == 'baye_fisher':
    from core import baye_fisher as approach
elif args.approach == 'sgd':
    from approaches import sgd as approach
elif args.approach == 'sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach == 'sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach == 'sgd_with_log':
    from approaches import sgd_with_log as approach
elif args.approach == 'sgd_L2_with_log':
    from approaches import sgd_L2_with_log as approach
elif args.approach == 'lwf':
    from approaches import lwf as approach
elif args.approach == 'lwf_with_log':
    from approaches import lwf_with_log as approach
elif args.approach == 'lfl':
    from approaches import lfl as approach
elif args.approach == 'ewc':
    from approaches import ewc as approach
elif args.approach == 'ewc_with_log':
    from approaches import ewc_with_log as approach
elif args.approach == 'si_with_log':
    from approaches import si_with_log as approach
elif args.approach == 'imm-mean':
    from approaches import imm_mean as approach
elif args.approach == 'imm-mode':
    from approaches import imm_mode as approach
elif args.approach == 'progressive':
    from approaches import progressive as approach
elif args.approach == 'pathnet':
    from approaches import pathnet as approach
elif args.approach == 'hat-test':
    from approaches import hat_test as approach
elif args.approach == 'hat':
    from approaches import hat as approach
elif args.approach == 'joint':
    from approaches import joint as approach

# Args -- Network
if args.approach == 'hat' or args.approach == 'hat-test':
    if args.conv_net:
        from core import conv_net_hat as network
    else:
        from networks import mlp_hat as network
elif args.approach == 'baye':
    if args.conv_net:
        from core import conv_networks as network
    else:
        from core import networks as network
else:
    if args.conv_net:
        from networks import conv_net as network
    else:
        from networks import mlp as network


########################################################################################################################

# Load
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed, tasknum=args.tasknum)
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
# print (inputsize,taskcla)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if args.approach == 'baye' and args.conv_net == False:
    net = network.BayesianNetwork(inputsize, taskcla, init_type='random',unitN=args.unitN, split = split, 
                                  notMNIST=notMNIST).cuda()
    net_old = network.BayesianNetwork(inputsize, taskcla, init_type='zero',unitN=args.unitN, split = split, 
                                      notMNIST=notMNIST).cuda()
    appr = approach.Appr(net, net_old, sbatch=args.batch_size, nepochs=args.nepochs, args=args, log_name=log_name, split=split)

elif args.approach == 'baye' and args.conv_net == True:
    net = network.BayesianConvNetwork(inputsize, taskcla, init_type='random').cuda()
    net_old = network.BayesianConvNetwork(inputsize, taskcla, init_type='zero').cuda()
    appr = approach.Appr(net, net_old, sbatch=args.batch_size, nepochs=args.nepochs, args=args, log_name=log_name, split=split)
    
else:
    if args.conv_net == False:
        net = network.Net(inputsize, taskcla, unitN=args.unitN, split = split, notMNIST=notMNIST).cuda()
        net_old = network.Net(inputsize, taskcla, unitN=args.unitN, split = split, notMNIST=notMNIST).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, nepochs=args.nepochs, args=args, log_name=log_name, split=split)
    else:
        net = network.Net(inputsize, taskcla).cuda()
        net_old = network.Net(inputsize, taskcla).cuda()
        appr = approach.Appr(net, sbatch=args.batch_size, nepochs=args.nepochs, args=args, log_name=log_name, split=split)

    
utils.print_model_report(net)

print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    if t == args.tasknum:
        break
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t == 0:
            xtrain = data[t]['train']['x']
            ytrain = data[t]['train']['y']
            xvalid = data[t]['valid']['x']
            yvalid = data[t]['valid']['y']
            task_t = t * torch.ones(xtrain.size(0)).int()
            task_v = t * torch.ones(xvalid.size(0)).int()
            task = [task_t, task_v]
        else:
            xtrain = torch.cat((xtrain, data[t]['train']['x']))
            ytrain = torch.cat((ytrain, data[t]['train']['y']))
            xvalid = torch.cat((xvalid, data[t]['valid']['x']))
            yvalid = torch.cat((yvalid, data[t]['valid']['y']))
            task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int()))
            task = [task_t, task_v]
    else:
        # Get data
        xtrain = data[t]['train']['x'].cuda()
        ytrain = data[t]['train']['y'].cuda()
        xvalid = data[t]['valid']['x'].cuda()
        yvalid = data[t]['valid']['y'].cuda()
        task = t

    # Train
    appr.train(task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
    print('-' * 100)

    # Test
    for u in range(t + 1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                      100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    # Save
    
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')
    torch.save(net.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))

# Done
print('*' * 100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t', end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100 * acc[i, j]), end='')
    print()
print('*' * 100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        # save task names
        from copy import deepcopy

        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t, ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t] = deepcopy(acc[t, :])
            appr.logs['test_loss'][t] = deepcopy(lss[t, :])
        # pickle
        import gzip
        import pickle

        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################

