#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :loss.py
# @Time      :2024/4/2 17:21
# @Author    :Luni Hu
import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets, ):
        effective_target = torch.eye(self.num_classes)[targets]
        # Calculate Cross entropy
        logit = F.softmax(inputs, dim=1)
        logit = logit.clamp(1e-7, 1.0)
        ce = -(effective_target * torch.log(logit))

        # Calculate Focal Loss
        weight = torch.pow(-logit + 1., self.gamma)
        fl = ce * weight * self.alpha

        if self.reduction == 'sum':
            return fl.sum()
        elif self.reduction == 'mean':
            return fl.mean()
