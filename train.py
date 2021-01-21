# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from models.liif import *
from models.misc import *
import utils
from test import eval_psnr
from configs.config_argument import train_arguments
from datasets.datasets import *


def prepare_training(train_args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if train_args.model == 'liif':
        model = LIIF(train_args)
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=train_args.lr)
    epoch_start = 1
    if train_args.scheduler == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    elif train_args.scheduler == 'MultiStepLR':
        if train_args.use_augmentation:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 100, 150, 180],
                                                          gamma=0.1)
        else:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.1)
    else:
        lr_scheduler = None
        # raise ValueError('Invalid scheduler')

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    inp_data_norm = train_args.inp_data_norm
    inp_sub, inp_div = list(map(float, inp_data_norm.split(',')))
    inp_sub = torch.FloatTensor([inp_sub]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor([inp_div]).view(1, -1, 1, 1).cuda()
    gt_data_norm = train_args.gt_data_norm
    gt_sub, gt_div = list(map(float, gt_data_norm.split(',')))
    gt_sub = torch.FloatTensor([gt_sub]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor([gt_div]).view(1, 1, -1).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'])

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None

    return train_loss.item()


def main(train_args, save_path):
    global log, writer
    log, writer = utils.set_save_path(save_path)

    train_loader, val_loader = make_data_loaders(train_args, log)

    model, optimizer, epoch_start, lr_scheduler = prepare_training(train_args)

    gpu_ids = [int(i) for i in train_args.gpu.split(',')]
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    epoch_max = train_args.epochs
    epoch_val = train_args.epoch_val
    epoch_save = train_args.epoch_save
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if len(gpu_ids) > 1:
            model_ = model.module
        else:
            model_ = model

        sv_file = {
            'model': model_.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if len(gpu_ids) > 1 and (train_args.eval_bsize is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_, inp_data_norm=train_args.inp_data_norm,
                                gt_data_norm=train_args.gt_data_norm,
                                eval_type=train_args.eval_type, eval_bsize=train_args.eval_bsize)

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':

    train_args = train_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = train_args.gpu
    save_path = os.path.join(train_args.save_path, train_args.model)

    main(train_args, save_path)
