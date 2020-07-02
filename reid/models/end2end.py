from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
import math

from .resnet import *
from .layers import LocalGlobalClassifier

__all__ = ["End2End_LocalGlobal",]


class End2End_LocalGlobal(nn.Module):
    def __init__(self, pretrained=True, dropout=0, num_classes=0):
        super(self.__class__, self).__init__()
        self.CNN = resnet50(dropout=dropout)
        self.lg_classifier = LocalGlobalClassifier(input_feature_size=2048, num_classes=num_classes, dropout=dropout)


    def forward(self, x):
        assert len(x.data.shape) == 5
        # reshape (batch, samples, ...) ==> (batch * samples, ...)
        oriShape = x.data.shape  # (batch, frames, 3, 256, 128)
        x = x.view(-1, oriShape[2], oriShape[3], oriShape[4])

        # resnet encoding
        resnet_feature = self.CNN(x)# (batch*frames, 2048)
        
        # reshape back into (batch, samples, ...)
        resnet_feature = resnet_feature.view(oriShape[0], oriShape[1], -1)  #(batch, frames, fea_dim)
        
        # classifer
        predict = self.lg_classifier(resnet_feature)

        return predict
