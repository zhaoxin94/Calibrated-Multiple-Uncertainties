import os
import os.path as osp
import argparse
import hashlib


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        "-d",
        default="officehome",
        help="Dataset",
        choices=['office31', 'officehome', 'visda', 'domainnet'])
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")

    parser.add_argument("--n_trials",
                        "-n",
                        default=3,
                        type=int,
                        help="Repeat times")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--mode', type=str, default='OPDA')

    # do not need to modify
    parser.add_argument("--method", "-m", default="OVA", help="Method")
    parser.add_argument("--backbone",
                        "-b",
                        default="resnet50",
                        help="Backbone")

    args = parser.parse_args()

    ##################################################################################

    data_root = '/home/zhao/data/DA'

    if args.dataset == 'office31':
        domains = ["amazon", "dslr", "webcam"]
        n_total = 31
        if args.mode == 'OPDA':
            n_share = 10
            n_source_private = 10
        elif args.mode == 'ODA':
            n_share = 10
            n_source_private = 0
    elif args.dataset == 'officehome':
        domains = ['art', 'clipart', 'product', 'real_world']
        n_total = 65
        if args.mode == 'OPDA':
            n_share = 10
            n_source_private = 5
        elif args.mode == 'ODA':
            n_share = 25
            n_source_private = 0
    elif args.dataset == 'visda':
        domains = ["synthetic", "real"]
        n_total = 12
        if args.mode == 'OPDA':
            n_share = 6
            n_source_private = 3
        elif args.mode == 'ODA':
            n_share = 6
            n_source_private = 0
    elif args.dataset == 'domainnet':
        n_total = 345
        domains = ['painting', 'real', 'sketch']
        if args.mode == 'OPDA':
            n_share = 150
            n_source_private = 50
    else:
        raise ValueError('Unknown Dataset: {}'.format(args.dataset))

    exp_info = args.exp_name
    if exp_info:
        exp_info = '_' + exp_info

    base_dir = osp.join('output', args.method, args.dataset + '_' + args.mode,
                        args.backbone + exp_info)

    for i in range(args.n_trials, args.n_trials + 1):
        for source in domains:
            for target in domains:
                if source != target:
                    if args.dataset == 'visda' and source == 'real':
                        print('skip real!')
                        continue
                    output_dir = osp.join(base_dir, source + '_to_' + target,
                                          str(i + 1))
                    seed = args.seed
                    if args.seed < 0:
                        seed = seed_hash(args.method, args.backbone,
                                         args.dataset, source, target, i)

                    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
                              f'python new/train_amp.py '
                              f'--root {data_root} '
                              f'--dataset {args.dataset} '
                              f'--source {source} '
                              f'--target {target} '
                              f'--n_share {n_share} '
                              f'--n_source_private {n_source_private} '
                              f'--n_total {n_total} '
                              f'--seed {seed}')
