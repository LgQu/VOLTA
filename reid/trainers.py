from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter
import numpy as np
import math


class BaseTrainer(object):
    def __init__(self, model, criterion, beta):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.beta = beta

    def train(self, epoch, data_loader, optimizer, print_freq=30):
        self.model.train()

        # The following code is used to keep the BN on the first three block fixed 
        fixed_bns = []
        for idx, (name, module) in enumerate(self.model.module.named_modules()):
            if name.find("layer3") != -1:
                assert len(fixed_bns) == 22
                break
            if name.find("bn") != -1:
                fixed_bns.append(name)
                module.eval() 

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        ce_losses = AverageMeter()
        info_losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)
            loss, prec = self._forward(inputs, targets, epoch)
            ce_loss, info_loss, loss = loss
            ce_losses.update(ce_loss.item(), inputs.size(0))
            info_losses.update(info_loss.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            precisions.update(prec, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            #torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.75)
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    'CE_Loss {:.3f} ({:.3f})\t'
                    'Info_Loss {:.3f} ({:.3f})\t'
                    'Prec {:.2%} ({:.2%})\t'
                    .format(epoch, i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg,
                            ce_losses.val, ce_losses.avg,
                            info_losses.val, info_losses.avg,
                            precisions.val, precisions.avg))


    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, epoch):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, vid_pids, _, _ = inputs # imgs.shape = (batch_size, frames, c, h, w)
        inputs = Variable(imgs, requires_grad=False)
        tmp_ones = torch.ones(imgs.size(1)).long()
        img_pids = torch.tensor([], dtype=torch.long)
        for i in range(vid_pids.numel()):
            img_pids = torch.cat([img_pids, tmp_ones * vid_pids[i]], dim=0)

        targets = [vid_pids.cuda(), img_pids.cuda()]    
        return inputs, targets

    def _forward(self, inputs, targets, epoch):
        outputs = self.model(inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            targets = torch.cat(targets, dim=0)
            (mu, std), logit = outputs
            ce_loss = self.criterion(logit, targets)
            info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))
            total_loss = ce_loss + self.beta * info_loss
            loss = (ce_loss, info_loss, total_loss)
            prec, = accuracy(logit.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec

