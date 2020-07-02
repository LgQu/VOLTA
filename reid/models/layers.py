from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
import math

class VIB(nn.Module): 
    # Variational Information Bottleneck 
    def __init__(self, input_feature_size, embeding_fea_size):
        super(self.__class__, self).__init__()
        self.encoder_mu = nn.Linear(input_feature_size, embeding_fea_size)
        self.encoder_std = nn.Linear(input_feature_size, embeding_fea_size)

        init.kaiming_normal_(self.encoder_mu.weight, mode='fan_out')
        init.constant_(self.encoder_mu.bias, 0)
        init.kaiming_normal_(self.encoder_std.weight, mode='fan_out')
        init.constant_(self.encoder_std.bias, 0)


    def forward(self, x, num_sample=1):
        mu = self.encoder_mu(x) 
        std = F.softplus(self.encoder_std(x)-5, beta=1) 
        encoding = self.reparametrize_n(mu,std,num_sample)
        return (mu, std), encoding

    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = std.data.new(std.size()).normal_()
        return mu + eps * std

class LocalGlobalClassifier(nn.Module):
    def __init__(self, input_feature_size, num_classes, embeding_fea_size=2048, dropout=0.5):
        super(self.__class__, self).__init__()

        # embeding
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        init.kaiming_normal_(self.embeding.weight, mode='fan_out')
        init.constant_(self.embeding.bias, 0)
        init.constant_(self.embeding_bn.weight, 1)
        init.constant_(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)


        self.vib = VIB(input_feature_size, embeding_fea_size)

        # classifier
        self.classify_fc = nn.Linear(embeding_fea_size, num_classes)
        init.normal_(self.classify_fc.weight, std = 0.001)
        init.constant_(self.classify_fc.bias, 0)

    def forward(self, inputs):
        """
        Args:
            inputs -- image-based feature, shape=(BS, frames, fea_dim)
        """
        vid_feat = inputs.mean(dim = 1) # (BS, fea_dim)
        batch_size, num_frame, fea_dim = inputs.size()
        img_feat = inputs.view(-1, fea_dim)
        all_feat = torch.cat([vid_feat, img_feat], dim=0)   #(BS+BS*num_frame, fea_dim)
        (mu, std), encoding = self.vib(all_feat)
        if (not self.training):
            vid_fc_feat = mu[:batch_size]
            img_fc_feat = mu[batch_size:].view(batch_size, num_frame, -1)
            return vid_fc_feat, img_fc_feat, img_feat.view(batch_size, num_frame, -1)

        # classifier
        logit = self.classify_fc(encoding)
        predict = (mu, std), logit
        return predict
