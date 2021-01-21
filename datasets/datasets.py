# -*- coding: utf-8 -*-

from .image_folder import *
from .wrappers import *
from torch.utils.data import DataLoader


def Data_load_train(root_path, repeat, cache, inp_size, scale_min, scale_max, augment, sample_q):
    dataset_use = ImageFolder(root_path=root_path, repeat=repeat, cache=cache)
    datasets_load = SRImplicitDownsampled(dataset_use, inp_size=inp_size, scale_min=scale_min, scale_max=scale_max,
                                          augment=augment, sample_q=sample_q)
    return datasets_load


def Data_load_val(root_path, first_k, repeat, cache, inp_size, scale_min, scale_max, sample_q):
    dataset_use = ImageFolder(root_path=root_path, first_k=first_k, repeat=repeat, cache=cache)
    datasets_load = SRImplicitDownsampled(dataset_use, inp_size=inp_size, scale_min=scale_min, scale_max=scale_max,
                                          sample_q=sample_q)
    return datasets_load


def Data_load_test(root_path_lr, root_path_hr):
    dataset_use = PairedImageFolders(root_path_1=root_path_lr, root_path_2=root_path_hr)
    datasets_load = SRImplicitPaired(dataset_use)
    return datasets_load


def make_data_loader(args_use, log, tag=''):
    if args_use is None:
        return None
    if tag == 'train':
        dataset = Data_load_train(args_use.train_path, args_use.train_repeat, args_use.train_cache, args_use.inp_size,
                                  args_use.scale_min, args_use.scale_max, args_use.augment, args_use.sample_q)
    else:
        dataset = Data_load_val(args_use.val_path, args_use.first_k_val, args_use.val_repeat, args_use.val_cache,
                                args_use.inp_size, args_use.scale_min, args_use.scale_max, args_use.sample_q)

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))
    loader = DataLoader(dataset, batch_size=args_use.batch_size,
                        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders(train_args, log):
    train_loader = make_data_loader(train_args, log, tag='train')
    val_loader = make_data_loader(train_args, log, tag='val')
    return train_loader, val_loader