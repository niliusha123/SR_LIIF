import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.liif import LIIF
import utils
from configs.config_argument import *
from datasets.datasets import *


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, inp_data_norm=None, gt_data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if inp_data_norm is None:
        inp_data_norm = '0, 1'
    if gt_data_norm is None:
        gt_data_norm = '0, 1'
    inp_data_norm = inp_data_norm
    inp_sub, inp_div = list(map(float, inp_data_norm.split(',')))
    inp_sub = torch.FloatTensor([inp_sub]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor([inp_div]).view(1, -1, 1, 1).cuda()
    gt_data_norm = gt_data_norm
    gt_sub, gt_div = list(map(float, gt_data_norm.split(',')))
    gt_sub = torch.FloatTensor([gt_sub]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor([gt_div]).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None:  # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    test_arguments = test_arguments()
    train_arguments = train_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = test_arguments.gpu

    dataset = Data_load_test(test_arguments.test_path_LR, test_arguments.test_path_HR)
    loader = DataLoader(dataset, batch_size=test_arguments.test_batch_size, num_workers=8, pin_memory=True)

    model = LIIF(train_arguments)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if test_arguments.gpu is not None:
        gpu_ids = [int(i) for i in test_arguments.gpu.split(',')]
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            model = model.to(device)
    model_spec = torch.load(test_arguments.model_save_dir)['model']
    if len(gpu_ids) > 1:
        model_ = model.module
    else:
        model_ = model
    model_.load_state_dict(model_spec)

    res = eval_psnr(loader, model, inp_data_norm=test_arguments.inp_data_norm,
                    gt_data_norm=test_arguments.gt_data_norm, eval_type=test_arguments.eval_type,
                    eval_bsize=test_arguments.eval_bsize, verbose=True)
    print('result: {:.4f}'.format(res))
