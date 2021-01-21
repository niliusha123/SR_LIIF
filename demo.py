import argparse
import os
import cv2

import torch
import numpy as np

import models
from models.liif import LIIF
from utils import make_coord
from test import batched_predict
from configs.config_argument import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./load/div2k/DIV2K_valid_LR_bicubic/X2/0801x2.png')
    parser.add_argument('--model', type=str, default='./save/edsr_baseline/epoch-best.pth')
    parser.add_argument('--resolution', default='480, 640')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = cv2.imread(args.input, 1) / 255
    img = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1)
    # img = torch.unsqueeze(torch.from_numpy(img).to(torch.float32), 0)

    model = LIIF(train_arguments())
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.gpu is not None:
        gpu_ids = [int(i) for i in args.gpu.split(',')]
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            model = model.to(device)
    model_spec = torch.load(args.model)['model']
    if len(gpu_ids) > 1:
        model_ = model.module
    else:
        model_ = model
    model_.load_state_dict(model_spec)
    model.eval()

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).cpu().numpy()
    # pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 1).cpu().numpy()
    # pred = (pred * 255).astype(np.uint8)
    pred = (pred * 255).astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output, pred)
