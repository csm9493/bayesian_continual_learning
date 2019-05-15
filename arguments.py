import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='', type=str, required=True,
                        choices=['mnist2', 'pmnist', 'col_pmnist', 'row_pmnist', 'mixture', 'omniglot','split_mnist','split_notmnist', 'split_cifar100'], help='(default=%(default)s)')
    parser.add_argument('--approach', default='', type=str, required=True,
                        choices=['random', 'sgd', 'sgd-frozen', 'sgd_with_log', 'sgd_L2_with_log', 'lwf',
                                 'lwf_with_log', 'lfl', 'ewc', 'ewc_with_log', 'baye', 'baye_fisher','baye_hat', 'imm-mean', 'progressive', 'pathnet',
                                 'imm-mode', 'sgd-restart', 'joint', 'hat', 'hat-test'], help='(default=%(default)s)')
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=50, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--unitN', default=400, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.05, type=float, required=False, help='(default=%(default)f)')
    parser.add_argument('--lamb', default='1', type=float, help='(default=%(default)f)')
    parser.add_argument('--beta', default='0.03', type=float, help='(default=%(default)f)')
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--conv-net', action='store_true', default=False, help='Using convolution network')
    parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
    parser.add_argument('--sample', type = int, default=1, help='Using sigma max to support coefficient')
    parser.add_argument('--dev-num', type = int, default=0, help='Device number')
    parser.add_argument('--drop', action='store_true', default=False, help='Use dropout')
    parser.add_argument('--rho', type = float, default=-2.783, help='initial rho')
    
    args = parser.parse_args()
    return args