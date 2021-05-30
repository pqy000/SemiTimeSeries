# -*- coding: utf-8 -*-

import torch
from optim.pytorchtools import EarlyStopping
import torch.nn as nn
from itertools import cycle
from collections import defaultdict
from torch.nn import functional as F
import numpy as np

class Model_SemiMean(torch.nn.Module):

    def __init__(self, model, ema_model, opt, feature_size=64, nb_class=3):
        super(Model_SemiMean, self).__init__()
        self.model = model
        self.ema_model = ema_model
        self.ce_loss = torch.nn.CrossEntropyLoss()
        # self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.epoch = 0
        self.usp_weight  = opt.usp_weight
        self.rampup = exp_rampup(opt.weight_rampup)

    def run_test(self, predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)

    def train(self, tot_epochs, train_loader, train_loader_label, val_loader, test_loader, opt):
        #### Training procedure #####
        patience = opt.patience
        early_stopping = EarlyStopping(patience, verbose=True,
                                       checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.learning_rate)
        train_max_epoch, train_best_acc, val_best_acc = 0, 0, 0
        for epoch in range(tot_epochs):
            self.model.train()
            self.ema_model.train()
            acc_label, acc_unlabel, loss_label, loss_unlabel = list(), list(), list(), list()
            for i, (data_labeled, data_unlabel) in enumerate(zip(cycle(train_loader_label), train_loader)):
                self.global_step += 1
                # label_data and # unlabel_data
                x, targets = data_labeled
                aug1, aug2, targetAug = data_unlabel
                x, targets, aug1, aug2, targetAug = x.cuda(), targets.cuda(), aug1.float().cuda(), \
                                                    aug2.float().cuda(), targetAug.cuda()
                # supervised loss
                outputs = self.model(x)
                loss = self.ce_loss(outputs, targets)
                prediction = outputs.argmax(-1)
                correct = prediction.eq(targets.view_as(prediction)).sum()
                loss_label.append(loss.item())
                acc_label.append(100.0 * correct / len(targets))

                # unsupervised consistency loss
                output_aug = self.model(aug1)
                self.update_ema(self.model, self.ema_model, opt.ema_decay, self.global_step)
                with torch.no_grad():
                    ema_aug = self.ema_model(aug2)
                    ema_aug = ema_aug.detach()
                cons_loss = mse_with_softmax(ema_aug, output_aug)
                cons_loss *= self.rampup(epoch) * self.usp_weight
                loss += cons_loss

                prediction = output_aug.argmax(-1)
                correct = prediction.eq(targetAug).sum()
                loss_unlabel.append(cons_loss.item())
                acc_unlabel.append(100.0 * correct / len(targetAug))

                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc_epoch_label = sum(acc_label) / len(acc_label)
            acc_epoch_unlabel = sum(acc_unlabel) / len(acc_unlabel)
            loss_epoch_unlabel = sum(loss_unlabel) / len(loss_unlabel)

            if acc_epoch_unlabel > train_best_acc:
                train_best_acc = acc_epoch_unlabel
                train_max_epoch = epoch

            acc_vals, acc_tests = list(), list()
            self.model.eval()
            with torch.no_grad():
                for i, (x, target) in enumerate(val_loader):
                    x, target = x.cuda(), target.cuda()
                    output = self.model(x)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_vals.append(accuracy.item())
                val_acc = sum(acc_vals) / len(acc_vals)

                if val_acc >= val_best_acc:
                    val_best_acc = val_acc
                    val_best_epoch = epoch
                    for i, (x, target) in enumerate(test_loader):
                        x, target = x.cuda(), target.cuda()
                        output = self.model(x)
                        prediction = output.argmax(-1)
                        correct = prediction.eq(target.view_as(prediction)).sum()
                        accuracy = (100.0 * correct / len(target))
                        acc_tests.append(accuracy.item())
                    test_acc = sum(acc_tests) / len(acc_tests)

                    print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'
                          .format(epoch, val_acc, test_acc, val_best_epoch))

            early_stopping(val_acc, self.model)
            if(early_stopping.early_stop):
                print("Early stopping")
                break

            if (epoch + 1) % opt.save_freq == 0:
                print("[INFO] save backbone at epoch {}!".format(epoch))
                torch.save(self.model.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))
            print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch Label ACC.= {:.2f}%, UnLabel ACC.= {:.2f}%, '
                  'Train Unlabel Best ACC.= {:.1f}%, Train Max Epoch={}' \
                  .format(epoch + 1, opt.model_name, opt.dataset_name, loss_epoch_unlabel,
                          acc_epoch_label, acc_epoch_unlabel,  train_best_acc, train_max_epoch))

        return test_acc, acc_epoch_unlabel, val_best_epoch

    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step +1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)

def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper

def mse_with_softmax(logit1, logit2):
    assert logit1.size() == logit2.size()
    return F.mse_loss(F.softmax(logit1, 1), F.softmax(logit2, 1))