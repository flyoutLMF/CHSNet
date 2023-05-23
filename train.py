import wandb
import argparse
import os
import torch
import numpy as np
import random

from utils.fsc_trainer import FSCTrainer

#  add test fsc

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--tag', default='chsnet', help='tag of training')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--no-wandb',  action='store_true', default=True, help='whether to use wandb')

    parser.add_argument('--data-dir', default=r'./datasets/FSC', help='training data directory')
    parser.add_argument('--log-param', type=float, default=100.0, help='dmap scale factor')
    parser.add_argument('--crop-size', type=int, default=384, help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16, help='downsample ratio')
    parser.add_argument('--dcsize', type=int, default=8, help='divide count size for density map')

    parser.add_argument('--lr', type=float, default=4*1e-5, help='the initial learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='train batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='the num of training process')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--max-epoch', type=int, default=1000, help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5, help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=50, help='the epoch start to val')

    parser.add_argument('--scheduler', type=str, default='step', help='or cosine')
    parser.add_argument('--step', type=int, default=400)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--t-max', type=int, default=200, help='for consine scheduler')
    parser.add_argument('--eta-min', type=float, default=4*1e-6, help='for consine scheduler')

    parser.add_argument('--save-dir', default='./checkpoint', help='directory to save models.')
    parser.add_argument('--save-all', type=bool, default=False, help='whether to save all best model')
    parser.add_argument('--max-model-num', type=int, default=1, help='max models num to save ')
    parser.add_argument('--resume', default='', help='the path of resume training model')

    parser.add_argument('--warm-up', type=int, default=-1, help='the warm up epoch num')
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for NCE')
    parser.add_argument('--samples', type=bool, default=True, help='get non-local samples from saved .npy')
    parser.add_argument('--lambda-nce', type=float, default=1e-4, help='weight for NCE loss')
    parser.add_argument('--sample-method', default='NonLocal', help='the sample method for negative samples,'
                                                                    ' select from ["NonLocal", "Random", "OnesMap"]')
    parser.add_argument('--feat-from', default='Attention', help='where the feature from, select from'
                                                                 ' ["Encoder", "Attention"]')
    parser.add_argument('--feat-get-method', default='MaxPool', help='the method to extract feature in a sample box,'
                                                                   ' select form ["Point", "MaxPool"]')
    parser.add_argument('--num-query', default='cl', help='the num of query feature, select from ["cl", "all"]')
    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:  # reproducible but slow
        torch.backends.cudnn.benchmark = False  # false by default, slow
        torch.backends.cudnn.deterministic = True  # Whether to use deterministic convolution algorithm? false by default.
    else:  # fast
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    setup_seed(43)
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(project="FSC", name=args.tag, config=vars(args))
    trainer = FSCTrainer(args)
    trainer.setup()
    trainer.train()
    wandb.finish()
