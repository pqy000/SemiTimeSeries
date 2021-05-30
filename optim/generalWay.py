# -*- coding: utf-8 -*-

import torch
import utils.transforms as transforms
from dataloader.ucr2018 import *
import torch.utils.data as data
from model.models import *
from model.architecture import TimeConv
from torch.utils.data.sampler import SubsetRandomSampler
from model.trainer import Model_SemiMean
import numpy as np
from model.TCNmodel import TCN

def train_SemiMean(x_train, y_train, x_val, y_val, x_test, y_test, opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    seqLen = x_train.shape[1]

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    ### Different Types of train, validation and test loader.
    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    ##########################################################
    #########Various data augmentation transformation#########
    ##########################################################
    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])
    tensor_transform = transforms.ToTensor()

    #########################################
    #########Different torch dataset#########
    #########################################
    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform_label)
    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)
    train_set = MultiUCR2018_Forecast(data=x_train, targets=y_train, K=K,
                                      transform=train_transform,
                                      totensor_transform=1)

    #######################################
    #########Separate labeled data#########
    #######################################
    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    #######################################
    #########Generate data loader##########
    #######################################
    train_loader_label = torch.utils.data.DataLoader(train_set_labeled, batch_size=batch_size, sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    ##############################################
    #########Generate model architec##############
    ##############################################

    # model = TimeConv(nb_class=opt.nb_class).cuda()
    # ema_model = TimeConv(nb_class=opt.nb_class).cuda()

    if opt.model_select == "TCN":
        channel_sizes = [opt.nhid] * opt.levels
        kernel_size = opt.ksize
        model = TCN(input_size=1, output_size=opt.nb_class,
                    num_channels=channel_sizes, kernel_size=kernel_size,
                    dropout=opt.dropout, series=opt.saliency).cuda()
        ema_model = TCN(input_size=1, output_size=opt.nb_class,
                    num_channels=channel_sizes, kernel_size=kernel_size,
                    dropout=opt.dropout, series=opt.saliency).cuda()
    else:
        model = TimeConv(nb_class=opt.nb_class).cuda()
        ema_model = TimeConv(nb_class=opt.nb_class).cuda()

    ##############################################
    #########Generate Semi training ##############
    ##############################################

    trainer = Model_SemiMean(model, ema_model, opt, feature_size, opt.nb_class).cuda()
    torch.save(trainer.model.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))

    ################################################
    ### Semi-Supervised training procedure##########
    ################################################
    test_acc, acc_unlabel, best_epoch = trainer.train(tot_epochs=tot_epochs, train_loader=train_loader,
                                                      train_loader_label=train_loader_label,
                                                      val_loader=val_loader,
                                                      test_loader=test_loader,
                                                      opt=opt)
    torch.save(trainer.model.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return test_acc, acc_unlabel, best_epoch
