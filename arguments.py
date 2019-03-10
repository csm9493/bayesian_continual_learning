import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='', type=str, required=True,
                        choices=['mnist2', 'pmnist', 'pmnist2', 'pmnist2_task15', 'pmnist2_task50',
                                 'cifar', 'mixture', 'omniglot'], help='(default=%(default)s)')
    parser.add_argument('--approach', default='', type=str, required=True,
                        choices=['random', 'sgd', 'sgd-frozen', 'sgd_with_log', 'sgd_L2_with_log', 'lwf',
                                 'lwf_with_log', 'lfl', 'ewc', 'ewc_with_log', 'baye', 'baye_hat', 'imm-mean', 'progressive', 'pathnet',
                                 'imm-mode', 'sgd-restart', 'joint', 'hat', 'hat-test'], help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=200, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='5000.', type=float, help='(default=%(default)f)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--conv-net', action='store_true', default=False, help='Using convolution network')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    args = parser.parse_args()
    return args
