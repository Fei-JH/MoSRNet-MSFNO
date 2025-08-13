# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:23:41 2024 (JST)

@author: Jinghao FEI
"""
import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        """
        初始化TVLoss

        参数:
            weight (float): TV损失的权重
            reduction (str): 指定如何减少损失。可选 'mean' 或 'sum'。
        """
        super(TVLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, x):
        """
        前向传播

        参数:
            x (torch.Tensor): 模型的输出，形状为 (batch_size, output_dim)

        返回:
            torch.Tensor: 计算得到的TV损失
        """
        # 计算相邻元素的差值
        diff = torch.abs(x[:, 1:] - x[:, :-1])
        if self.reduction == 'mean':
            # 计算平均TV损失
            tv_loss = self.weight * torch.mean(diff)
        elif self.reduction == 'sum':
            # 计算总和TV损失
            tv_loss = self.weight * torch.sum(diff)
        else:
            # 不进行减少操作
            tv_loss = self.weight * diff
        return tv_loss