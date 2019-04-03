import sys, os, time
import numpy as np
import torch

import utils
from arguments import get_args

tstart = time.time()

# Arguments

args = get_args()
args_std = np.log(1+np.exp(args.rho))
log_name = '{}_{}_{}_{}_{}_lamb_{}_{}_{}_{}_{}'.format(args.date, args.experiment, args.tasknum, args.approach, args.seed,
                                              args.lamb, args.nepochs, args.sample, args.lr,args_std)
if args.use_sigmamax:
    log_name += '_sigmamax'

if args.use_Bernoulli:
    log_name += '_Bernoulli'

if args.use_Attention:
    log_name += '_Attention'

if args.use_Dropout:
    log_name += '_Dropout'

if args.no_sigma_reg:
    log_name += 'no_sigma_reg'
    
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
elif args.experiment == 'pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment == 'pmnist2':
    from dataloaders import pmnist2 as dataloader
elif args.experiment == 'pmnist2_task15':
    from dataloaders import pmnist2_task15 as dataloader
elif args.experiment == 'pmnist2_task50':
    from dataloaders import pmnist2_task50 as dataloader
elif args.experiment == 'cifar':
    from dataloaders import cifar as dataloader
elif args.experiment == 'mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach == 'random':
    from approaches import random as approach
elif args.approach == 'baye':
    from core import baye as approach
elif args.approach == 'baye_hat':
    from core import baye_hat as approach
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
if args.experiment == 'mnist2' or args.experiment == 'pmnist' or args.experiment == 'pmnist2' or args.experiment == 'pmnist2_task15' or args.experiment == 'pmnist2_task50':
    if args.approach == 'hat' or args.approach == 'hat-test':
        from networks import mlp_hat as network
    elif args.approach == 'baye' or args.approach == 'baye_hat' or args.approach == 'baye_fisher':
        if args.conv_net:
            from core import conv_network as network
        else:
            from core import networks as network
    else:
        if args.conv_net:
            from networks import conv_net as network
        else:
            from networks import mlp as network
else:
    if args.approach == 'lfl':
        from networks import alexnet_lfl as network
    elif args.approach == 'hat':
        from networks import alexnet_hat as network
    elif args.approach == 'progressive':
        from networks import alexnet_progressive as network
    elif args.approach == 'pathnet':
        from networks import alexnet_pathnet as network
    elif args.approach == 'hat-test':
        from networks import alexnet_hat_test as network
    else:
        from networks import alexnet as network

########################################################################################################################

# Load
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed)
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
# print (inputsize,taskcla)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
if args.approach == 'baye' or args.approach == 'baye_hat':
    net = network.BayesianNetwork(inputsize, taskcla, init_type='random', rho_init=args.rho, dropout = args.use_Dropout).cuda()
    net_old = network.BayesianNetwork(inputsize, taskcla, init_type='zero', rho_init=args.rho, dropout = args.use_Dropout).cuda()
    appr = approach.Appr(net, net_old, nepochs=args.nepochs, sample = args.sample, lr=args.lr, args=args, log_name=log_name)

else:
    net = network.Net(inputsize, taskcla).cuda()
    appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args, log_name=log_name)

utils.print_model_report(net)

print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
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
        if args.approach == 'baye' or args.approach == 'baye_hat' or args.approach == 'baye_fisher':
            test_loss, test_acc = appr.eval(xtest, ytest)
        else:
            test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                      100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss

    # Save
    
    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')
    """
    f = open('result_data/std_value.txt','a')
    min_arr = []
    min_idx_arr = []
    max_arr = []
    max_idx_arr = []
    mean, var, rho_sum = 0, 0, 0
    for (_, layer) in net.named_children():
        #rho = torch.log1p(torch.exp(layer.weight_rho))
        rho = layer.weight_rho
        rho = rho.data.cpu().numpy()
        min_arr.append(np.min(rho))
        min_idx_arr.append(np.argmin(rho))
        max_arr.append(np.max(rho))
        max_idx_arr.append(np.argmin(rho))
        mean = np.mean(rho)
        var = np.var(rho)
        rho_sum += np.sum(rho<0.065)
    f.write('TASK:%d\n'%t)
    
    f.write('minimum std:\n')
    for i in range(len(min_arr)):
        f.write('%f '%(min_arr[i]))
    
    f.write('min idx std:\n')
    for i in range(len(min_arr)):
        f.write('%d '%(min_idx_arr[i]))
    
    f.write('\n maximum std:\n')
    for i in range(len(min_arr)):
        f.write('%f '%(max_arr[i]))
    
    f.write('max idx std:\n')
    for i in range(len(min_arr)):
        f.write('%d '%(max_idx_arr[i]))
        
    f.write('\n mean: %f'%(mean))
    f.write('\n var: %f'%(var))
    f.write('\n sum: %d'%(rho_sum))
    
    f.write('\n')
    f.flush()
    """
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

