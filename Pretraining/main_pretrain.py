# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_lomar

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=75, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    # parser.add_argument('--model', default='mae_vit_tiny', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    # parser.add_argument('--model', default='mae_vit_small', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
    #                     help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='../dataset/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--init_ckpt', default='',
                        help='optional initialization checkpoint; leave empty to train from scratch')
    parser.add_argument('--max_train_steps', default=0, type=int,
                        help='limit batches per epoch for smoke tests; 0 means use the full epoch')
    parser.add_argument('--amp_dtype', default='bf16', choices=('bf16', 'fp16', 'none'),
                        help='mixed precision type; bf16 is recommended on A100')
    parser.add_argument('--synthetic_data', action='store_true',
                        help='use generated images for memory/smoke tests only')
    parser.add_argument('--synthetic_length', default=65536, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # lomar parameters
    parser.add_argument("--window_size",default=7,type=int)
    parser.add_argument("--num_window",default=4, type=int)
    parser.add_argument('--mask_ratio', default=0.8, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--lfst_cutoff', default=30, type=int)
    parser.add_argument('--lfst_loss_weight', default=0.3, type=float)
    parser.add_argument('--sasgt_scales', default='0.8,1.6,3.2,6.4', type=str)
    parser.add_argument('--sasgt_temperature', default=1.0, type=float)
    parser.add_argument('--sasgt_gamma', default=1.0, type=float)
    parser.add_argument('--sasgt_reliability_window', default=7, type=int)
    parser.add_argument('--save_freq', default=50, type=int,
                        help='checkpoint interval; <=0 saves only the final epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--MASTER_ADDR', default='localhost',
                        help='we use teh local host')
    parser.add_argument("--MASTER_PORT",default="10019")
    parser.add_argument("--distributed",action="store_true")

    return parser

# from torch.distributed.elastic.multiprocessing.errors import record

# @record
def main(args):
    from pathlib import Path

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size, scale=(0.2, 1.0),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    from util.datasets import SyntheticSARDataset, load_data
    if args.synthetic_data:
        dataset_train = SyntheticSARDataset(args.synthetic_length, args.input_size)
    else:
        dataset_train = load_data(os.path.join(args.data_path), transform=transform_train)

    # custom_dataset = load_data(os.path.join(args.data_path), transform=transform_train)
    # train_size = int(len(custom_dataset) * 0.4)
    # dataset_train, _ = torch.utils.data.random_split(custom_dataset, [train_size,  len(custom_dataset) - train_size])

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    
    # define the model
    sasgt_scales = tuple(float(value) for value in args.sasgt_scales.split(','))
    model = models_lomar.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        lfst_cutoff=args.lfst_cutoff,
        lfst_loss_weight=args.lfst_loss_weight,
        sasgt_scales=sasgt_scales,
        sasgt_temperature=args.sasgt_temperature,
        sasgt_gamma=args.sasgt_gamma,
        sasgt_reliability_window=args.sasgt_reliability_window,
    )

    if args.init_ckpt:
        if not os.path.isfile(args.init_ckpt):
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {args.init_ckpt}. "
                "Pass --init_ckpt '' to train from scratch."
            )
        checkpoint = torch.load(args.init_ckpt, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)


    model.to(device)

    # model = torch.nn.DataParallel(model, device_ids=[5,6,7])

    model_without_ddp = model
    trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(f"Model: {args.model}; trainable parameters: {trainable_params / 1e6:.2f}M")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    print(args.distributed)
    if args.distributed:
        print(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # with torch.autograd.detect_anomaly():
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        periodic_save = args.save_freq > 0 and (epoch + 1) % args.save_freq == 0
        if args.output_dir and (periodic_save or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
