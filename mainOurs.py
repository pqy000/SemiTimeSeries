# -*- coding: utf-8 -*-

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from optim.pretrain import *
from optim.generalWay import *
from optim.train import supervised_train
from datetime import datetime

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--save_freq', type=int, default=200,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--K', type=int, default=4, help='Number of augmentation for each sample')
    parser.add_argument('--alpha', type=float, default=0.5, help='Past-future split point')

    parser.add_argument('--feature_size', type=int, default=64,
                        help='feature_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='training patience')
    parser.add_argument('--aug_type', type=str, default='none', help='Augmentation type')

    parser.add_argument('--class_type', type=str, default='3C', help='Classification type')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_rampup', type=int, default=30, help='weight rampup')
    # model dataset
    parser.add_argument('--dataset_name', type=str, default='CricketX',
                        choices=['CricketX',
                                 'UWaveGestureLibraryAll',
                                 'InsectWingbeatSound',
                                 'MFPT', 'XJTU',
                                 'EpilepticSeizure',],
                        help='dataset')
    parser.add_argument('--nb_class', type=int, default=3, help='class number')

    # ucr_path = '../datasets/UCRArchive_2018'
    parser.add_argument('--ucr_path', type=str, default='./datasets',
                        help='Data root for dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Data path for checkpoint.')
    # method
    parser.add_argument('--backbone', type=str, default='SimConv4')
    parser.add_argument('--model_name', type=str, default='SemiTeacher',
                        choices=['SupCE', 'SemiTime','SemiTeacher'], help='choose method')
    parser.add_argument('--label_ratio', type=float, default=0.4, help='label ratio')
    parser.add_argument('--usp_weight', type=float, default=0.3, help='usp weight')
    parser.add_argument('--ema_decay', type=float, default=0.95, help='usp weight')
    parser.add_argument('--model_select', type=str, default='TCN', help='Training model type')
    parser.add_argument('--nhid', type=int, default=64, help='feature_size')
    parser.add_argument('--levels', type=int, default=8, help='feature_size')
    parser.add_argument('--ksize', type=int, default=3, help='kernel size')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (default: 0.05)')
    parser.add_argument('--saliency', type=bool, default=True, help='Training with series saliency')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    import os
    import numpy as np

    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    exp = 'exp-cls'

    Seeds = [0]
    Runs = range(0, 1, 1)

    aug1 = ['jitter','cutout']
    aug2 = ['G0']
    # datetime = str(datetime.now().strftime("%Y-%m-%d-%H\%M\%S"))
    if opt.model_name == 'SemiTime':
        model_paras = 'label{}_{}'.format(opt.label_ratio, opt.alpha)
    if opt.model_name == "SemiTeacher" and opt.saliency:
        model_paras = 'label{}_SS'.format(opt.label_ratio)
    else:
        model_paras = 'label{}'.format(opt.label_ratio)

    if aug1 == aug2:
        opt.aug_type = [aug1]
    elif type(aug1) is list:
        opt.aug_type = aug1 + aug2
    else:
        opt.aug_type = [aug1, aug2]

    log_dir = './results/{}/{}/{}/{}'.format(exp, opt.dataset_name, opt.model_name, model_paras)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file2print_detail_train = open("{}/train_detail.log".format(log_dir), 'a+')
    print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), file=file2print_detail_train)
    print("Dataset  Train  Test  Dimension  Class  Seed  Acc_label  Acc_unlabel  Epoch_max",file=file2print_detail_train)
    file2print_detail_train.flush()

    file2print = open("{}/test.log".format(log_dir), 'a+')
    print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), file=file2print)
    print("Dataset  Acc_mean   Acc_std  Epoch_max", file=file2print)
    file2print.flush()

    file2print_detail = open("{}/test_detail.log".format(log_dir), 'a+')
    print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), file=file2print_detail)
    print("Dataset  Train  Test   Dimension  Class  Seed  Acc_max  Epoch_max",
          file=file2print_detail)
    file2print_detail.flush()

    ACCs = {}

    MAX_EPOCHs_seed = {}
    ACCs_seed = {}
    for seed in Seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        opt.ckpt_dir = './ckpt/{}/{}/{}/{}/{}/{}'.format(
            exp, opt.model_name, opt.dataset_name, '_'.join(opt.aug_type),
            model_paras, str(seed))

        if not os.path.exists(opt.ckpt_dir):
            os.makedirs(opt.ckpt_dir)

        print('[INFO] Running at:', opt.dataset_name)

        x_train, y_train, x_val, y_val, x_test, y_test, opt.nb_class, _ = load_ucr2018(opt.ucr_path, opt.dataset_name)

        ACCs_run = {}
        MAX_EPOCHs_run = {}
        for run in Runs:

            if opt.model_name == 'SupCE':
                acc_test, epoch_max = supervised_train(
                    x_train, y_train, x_val, y_val, x_test, y_test, opt)
                acc_unlabel = 0

            elif 'SemiTime' in opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiTime(x_train, y_train, x_val, y_val, x_test, y_test, opt)

            elif 'SemiTeacher' in opt.model_name:
                acc_test, acc_unlabel, epoch_max = train_SemiMean(x_train, y_train, x_val, y_val, x_test, y_test, opt)

            print("{}  {}  {}  {}  {}  {}  {}  {}  {}".format(
                opt.dataset_name, x_train.shape[0], x_test.shape[0],
                x_train.shape[1], opt.nb_class, seed, acc_test, acc_unlabel, epoch_max),
                file=file2print_detail_train)
            file2print_detail_train.flush()

            ACCs_run[run] = acc_test
            MAX_EPOCHs_run[run] = epoch_max

        ACCs_seed[seed] = round(np.mean(list(ACCs_run.values())), 2)
        MAX_EPOCHs_seed[seed] = np.max(list(MAX_EPOCHs_run.values()))

        print("{}  {}  {}  {}  {}  {}  {}  {}".format(
            opt.dataset_name, x_train.shape[0], x_test.shape[0], x_train.shape[1], opt.nb_class,
            seed, ACCs_seed[seed], MAX_EPOCHs_seed[seed]),
            file=file2print_detail)

        file2print_detail.flush()

    ACCs_seed_mean = round(np.mean(list(ACCs_seed.values())), 2)
    ACCs_seed_std = round(np.std(list(ACCs_seed.values())), 2)
    MAX_EPOCHs_seed_max = np.max(list(MAX_EPOCHs_seed.values()))

    print("{} {} {} {}".format(opt.dataset_name, ACCs_seed_mean,
                               ACCs_seed_std, MAX_EPOCHs_seed_max), file=file2print)
    file2print.flush()
