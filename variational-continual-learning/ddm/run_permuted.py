import numpy as np
import tensorflow as tf
import gzip
import _pickle as cPickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description='Continual')
parser.add_argument('--date', default='', type=str, help='result date')
parser.add_argument('--experiment', default='pmnist', type=str, help='which experiment')
parser.add_argument('--trial', default=1, type=int, help='what numberth of result')
parser.add_argument('--batch', default=64, type=int, help='mini batch size')
parser.add_argument('--epochs', default=50, type=int, help='# of epochs to train')
parser.add_argument('--tasknum', default=10, type=int, help='# of tasks to train')


args = parser.parse_args()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# tf.keras.backend.set_session(session)

# gpu_options = tf.GPUOptions(visible_device_list = '0')
# config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True,device_count={'GPU':1}, gpu_options=gpu_options)
# config.gpu_options.per_process_gpu_memory_fraction = 0.3

# session = tf.Session(config=config)
# tf.keras.backend.set_session(session)

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
#             np.random.seed(self.cur_iter)
#             perm_inds = range(self.X_train.shape[1])
#             np.random.shuffle(perm_inds)
            perm_inds = np.random.permutation(self.X_train.shape[1])
#             print(perm_inds[:10])
            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [400, 400]
batch_size = args.batch
no_epochs = args.epochs
single_head = True
num_tasks = args.tasknum
train_info = {}
train_info['date'] = args.date
train_info['experiment'] = args.experiment
train_info['trial'] = args.trial
train_info['batch'] = args.batch
train_info['tasknum'] = args.tasknum

# Run vanilla VCL
tf.set_random_seed(11 + args.trial)
np.random.seed(args.trial)

train_info['coreset_method']='none'
coreset_size = 0
data_gen = PermutedMnistGenerator(num_tasks)
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head, train_info)
print (vcl_result)

"""

# Run random coreset VCL
tf.reset_default_graph()
tf.set_random_seed(11 + args.trial)
np.random.seed(args.trial)

train_info['coreset_method']='random'
coreset_size = 200
data_gen = PermutedMnistGenerator(num_tasks)
rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.rand_from_batch, coreset_size, batch_size, single_head, train_info)
print (rand_vcl_result)

# Run k-center coreset VCL
tf.reset_default_graph()
tf.set_random_seed(11 + args.trial)
np.random.seed(args.trial)

train_info['coreset_method']='kcenter'
data_gen = PermutedMnistGenerator(num_tasks)
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.k_center, coreset_size, batch_size, single_head, train_info)
print (kcen_vcl_result)

# Plot average accuracy
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
utils.plot('results/permuted.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)

"""
