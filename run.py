import argparse
import os
import random

import numpy as np
import torch
from loguru import logger

import lthNet
from data.data_loader import load_data


def run():
    # Load config
    args = load_config()
    logger.add('logs/{}_model_{}_code_{}_beta_{}_gamma_{}_batchsize_{}.log'.format(
        args.dataset,
        args.arch,
        args.code_length,
        args.beta,
        args.gamma,
        args.batch_size,
    ),
        rotation='500 MB',
        level='INFO',
    )
    logger.info(args)

    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    train_dataloader, query_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.batch_size,
        args.num_workers,
    )

    # class-samples mapping (train_dataloader). eg., mapping['0']=500, mapping['1']=100, etc.
    class_samples = torch.Tensor(np.zeros(args.num_classes))
    for _, targets, _ in train_dataloader:
        class_samples += torch.sum(targets, dim=0)

    mapping = {}
    for i in range(len(class_samples)):
        mapping[str(i)] = class_samples[i]

    # Training
    for code_length in args.code_length:
        checkpoint = lthNet.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args.arch,
            args.feature_dim,
            code_length,
            args.num_classes,
            args.dynamic_meta_embedding,
            args.num_prototypes,
            args.device,
            args.lr,
            args.max_iter,
            args.beta,
            args.gamma,
            mapping,
            args.topk,
            args.evaluate_interval,
        )
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, checkpoint['map']))

        # Save checkpoint
        torch.save(
            checkpoint,
            os.path.join('checkpoints', '{}_model_{}_code_{}_beta_{}_gamma_{}_map_{:.4f}_batchsize_{}_maxIter_{}.pt'.format(
                args.dataset,
                args.arch,
                code_length,
                args.beta,
                args.gamma,
                checkpoint['map'],
                args.batch_size,
                args.max_iter)
                         )
        )


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='LTHNet_PyTorch_LinearLoss')
    parser.add_argument('--dataset', default='imagenet-100-IF1', type=str,
                        help='Dataset name.')
    parser.add_argument('--root', default='C:\\Users\\dell\\Desktop\\LTH_linearloss\\data\\imagenet100\\', type=str,
                        help='Path of dataset')
    # parser.add_argument('--root', default='/home/13810427976/notespace/cifar-100/cifar-100-IF20/', type=str,
    #                    help='Path of dataset')
    parser.add_argument('--code-length', default='32,48,64,96', type=str,
                        help='Binary hash code length.(default: 32,48,64,96)')
    parser.add_argument('--arch', default='resnet34', type=str,
                        help='CNN model name.(default: alexnet)')
    parser.add_argument('--feature-dim', default=2000, type=int,
                        help='number of classes.(default: 2000)')
    parser.add_argument('--num-classes', default=100, type=int,
                        help='number of classes.(default: 100)')
    parser.add_argument('--num-prototypes', default=100, type=int,
                        help='number of prototypes.(default: 100)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Learning rate.(default: 1e-5)')
    parser.add_argument('--max-iter', default=100, type=int,
                        help='Number of iterations.(default: 300)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--dynamic-meta-embedding', default=True, type=bool,
                        help='dynamic meta embedding.(default: True)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--beta', default=0.0, type=float,
                        help='Hyper-parameter: class-balanced factor.(default: 0.99)')
    parser.add_argument('--gamma', default=1.0, type=float,
                        help='Hyper-parameter: balance between pointwise and pairwise loss.(default: 1.0)')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')
    parser.add_argument('--evaluate-interval', default=1, type=int,
                        help='Evaluation interval.(default: 10)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.device = torch.device("cuda:%d" % 0)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))

    return args


if __name__ == '__main__':
    run()
