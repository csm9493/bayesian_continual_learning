import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='', type=str, required=True,
                        choices=['mnist2', 'pmnist', 'pmnist2', 'pmnist3', 'pmnist2_task15', 'pmnist2_task50',
                                 'cifar', 'mixture', 'omniglot'], help='(default=%(default)s)')
    parser.add_argument('--approach', default='', type=str, required=True,
                        choices=['random', 'sgd', 'sgd-frozen', 'sgd_with_log', 'sgd_L2_with_log', 'lwf',
                                 'lwf_with_log', 'lfl', 'ewc', 'ewc_with_log', 'baye', 'baye_fisher','baye_hat', 'imm-mean', 'progressive', 'pathnet',
                                 'imm-mode', 'sgd-restart', 'joint', 'hat', 'hat-test'], help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=50, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='0.1', type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default='0.03', type=float, help='(default=%(default)f)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--conv-net', action='store_true', default=False, help='Using convolution network')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=5, help='Using sigma max to support coefficient')
    parser.add_argument('--rho', type = float, default=-2.783, help='initial rho')
    args = parser.parse_args()
    return args





"""
parser.add_argument('--tau', default='30', type=float, help='(default=%(default)f)')
parser.add_argument('--gamma', default='0.001', type=float, help='(default=%(default)f)')
parser.add_argument('--use-sigmamax', action='store_true', default=False, help='Using sigma max to support coefficient')
parser.add_argument('--use-sigmamean', action='store_true', default=False, help='Using sigma mean to support coefficient')
parser.add_argument('--use-sigmainit', action='store_true', default=False, help='Using sigma init to support coefficient')
parser.add_argument('--use-divide', action='store_true', default=False, help='Using sigma mean to support coefficient')
parser.add_argument('--normal-reg', action='store_true', default=False, help='Use regularization with normal distribution')
parser.add_argument('--normal-std-reg', action='store_true', default=False, help='Use regularization with normal distribution and prev posterior')
parser.add_argument('--sparse', action='store_true', default=False, help='use group sparse')
parser.add_argument('--reg-max', action='store_true', default=False, help='use reg max')
parser.add_argument('--reg-min', action='store_true', default=False, help='use reg min')
parser.add_argument('--L1-min', action='store_true', default=False, help='use L1 reg min')
parser.add_argument('--L2-min', action='store_true', default=False, help='use L2 reg min')
parser.add_argument('--prune', action='store_true', default=False, help='use pruning loss')
parser.add_argument('--node-wise', action='store_true', default=False, help='use node-wise regularization')
parser.add_argument('--channel-wise', action='store_true', default=False, help='use channel-wise regularization')
parser.add_argument('--use-slow-update', action='store_true', default=False, help='update per 10% decrease of std')
parser.add_argument('--var-init', action='store_true', default=False, help='variance init per save')
"""