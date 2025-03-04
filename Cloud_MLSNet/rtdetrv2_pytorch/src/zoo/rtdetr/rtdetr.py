"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
from select import epoll

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR',]


# @register()
# class RTDETR(nn.Module):
#     __inject__ = ['backbone', 'encoder', 'decoder', ]
#
#     def __init__(self, \
#         backbone: nn.Module,
#         encoder: nn.Module,
#         decoder: nn.Module,
#     ):
#         super().__init__()
#         self.backbone = backbone
#         self.decoder = decoder
#         self.encoder = encoder
#
#     def forward(self, x, targets=None):
#         x = self.backbone(x)
#         x = self.encoder(x)
#         x = self.decoder(x, targets)
#
#         return x
#
#     def deploy(self, ):
#         self.eval()
#         for m in self.modules():
#             if hasattr(m, 'convert_to_deploy'):
#                 m.convert_to_deploy()
#         return self

@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
                 backbone: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        for param in self.backbone.parameters():
            param.requires_grad = False
        with torch.no_grad():
            x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

@register()
class RTDETR_freezebackbone_by_epoch(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
                 backbone: nn.Module,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.freeze_after_epoch = 130
        self.current_epoch = 0  # 当前训练轮次

    def forward(self, x, targets=None):
        # 使用 no_grad 来禁用梯度计算（如果 backbonen 冻结）
        if self.current_epoch >= self.freeze_after_epoch:
            for param in self.backbone.parameters():
                param.requires_grad = False
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)

        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def set_epoch_for_backbone(self, epoch):
        """设置当前训练轮次，用于决定是否冻结参数"""
        self.current_epoch = epoch

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

