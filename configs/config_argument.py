# -*- coding: utf-8 -*-

import argparse


def train_arguments():
    args = argparse.ArgumentParser()
    # data
    args.add_argument('--train_path', type=str, default='./load/div2k/DIV2K_train_HR')
    args.add_argument('--val_path', type=str, default='./load/div2k/DIV2K_valid_HR')
    args.add_argument('--train_repeat', type=int, default=20)
    args.add_argument('--val_repeat', type=int, default=160)
    args.add_argument('--train_cache', type=str, default='in_memory')
    args.add_argument('--val_cache', type=str, default='in_memory')
    args.add_argument('--inp_size', type=int, default=48)
    args.add_argument('--scale_min', type=int, default=1)
    args.add_argument('--scale_max', type=int, default=2)
    args.add_argument('--augment', type=bool, default=True)
    args.add_argument('--sample_q', type=int, default=2304)
    args.add_argument('--first_k_val', type=int, default=10)
    args.add_argument('--inp_data_norm', type=str, default='0.5, 0.5')
    args.add_argument('--gt_data_norm', type=str, default='0.5, 0.5')

    # Model parameters
    args.add_argument('--model', type=str, choices=['liif'], default='liif')
    args.add_argument('--encoder_spec', type=str, choices=['edsr', 'imdn', 'rcan', 'rdn', 'maffsrn'], default='rdn')
    args.add_argument('--no_upsampling', type=bool, default=True)
    args.add_argument('--local_ensemble', type=bool, default=True)
    args.add_argument('--feat_unfold', type=bool, default=True)
    args.add_argument('--cell_decode', type=bool, default=True)
    args.add_argument('--imnet_spec', type=str, default='mlp')
    args.add_argument('--out_dim', type=int, default=3)
    args.add_argument('--hidden_list', type=list, default=[256, 256, 256, 256])
    args.add_argument('--save_path', type=str, default='./save')

    # Training parameters
    args.add_argument('--epochs', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--scheduler', type=str, choices=['StepLR', 'MultiStepLR'], default='MultiStepLR')
    args.add_argument('--use_augmentation', type=bool, default=True)
    args.add_argument('--lr', type=float, default=1.e-4)
    args.add_argument('--gpu', type=str, default='3')
    args.add_argument('--epoch_val', type=int, default=1)
    args.add_argument('--epoch_save', type=int, default=100)
    args.add_argument('--eval_bsize', type=int, default=None)
    args.add_argument('--eval_type', type=str, default=None)

    return args.parse_args()


def test_arguments():
    args = argparse.ArgumentParser()
    # data
    args.add_argument('--test_path_LR', type=str, default='./load/div2k/DIV2K_valid_LR_bicubic/X2')
    args.add_argument('--test_path_HR', type=str, default='./load/div2k/DIV2K_valid_HR')
    args.add_argument('--model_save_dir', type=str, default='./save/edsr_baseline/epoch-best.pth')
    args.add_argument('--inp_data_norm', type=str, default='0.5, 0.5')
    args.add_argument('--gt_data_norm', type=str, default='0.5, 0.5')

    # Testing parameters
    args.add_argument('--gpu', type=str, default='1')
    args.add_argument('--eval_bsize', type=int, default=30000)
    args.add_argument('--eval_type', type=str, default='div2k-2')
    args.add_argument('--test_batch_size', type=int, default=1)

    return args.parse_args()

