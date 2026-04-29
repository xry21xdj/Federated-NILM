import os
import argparse


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    args_string = ""
    args_to_show = ["experiment", "method"]
    for arg in args_to_show:
        args_string = os.path.join(args_string, str(getattr(args, arg)))
    if args.locally_tune_clients:
        args_string += "_adapt"

    return args_string


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',help='name of experiment',type=str,default='emnist')
    parser.add_argument('--method', help='the method to be used;'' possible are `FedAvg`, FedMGDA-M, DOMO,',
                        type=str,default='FedAvg')
    parser.add_argument('--n_rounds',type=int, default=5, help='number of communication rounds; default is 1')
    parser.add_argument('--bz', type=int,default=128,help='batch_size,default is 128')
    parser.add_argument('--num_workers',help='number of workers for dataloader; default is 2',type=int,default=1)
    parser.add_argument('--local_steps',help='number of local steps before communication; default is 1',type=int,default=1)
    parser.add_argument('--log_freq',help='frequency of writing logs; defaults is 1',type=int,default=1)
    parser.add_argument('--device',help='device to use, either cpu or cuda; default is cpu',type=str,default="cpu")
    parser.add_argument('--optimizer',help='optimizer to be used for the training; default is sgd',type=str,default="sgd")
    parser.add_argument("--lr",type=float,help='learning rate; default is 1e-3', default=1e-3)
    parser.add_argument("--lr_scheduler",help='learning rate decay scheme to be used;'' possible are "sqrt", "linear", "cosine_annealing", "multi_step" and "constant" (no learning rate decay);'
                        'default is "constant"',type=str,default="constant")
    parser.add_argument("--mu",help='proximal / penalty term weight, used when --optimizer=`prox_sgd` also used with L2SGD; default is `0.`',
                        type=float,default=0)
    parser.add_argument("--gamma",help='momentum coefficient for FedNAG default is `0.`',type=float, default=1)
    parser.add_argument("--beta",help='weight for the FedCM or simi-FedCM method',type=float,default=0)
    parser.add_argument("--seed",help='random seed',type=int,default=1234)

    parser.add_argument(
        "--locally_tune_clients",
        help='if selected, clients are tuned locally for one epoch before writing logs;',
        action='store_true'
    )
    parser.add_argument(
        '--validation',
        help='if chosen the validation part will be used instead of test part;'
             ' make sure to use `val_frac > 0` in `generate_data.py`;',
        action='store_true'
    )
    
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args