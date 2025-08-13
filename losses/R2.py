# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:08:20 2024 (JST)

@author: Jinghao FEI
"""

import torch

class R2(object):
    def __init__(self, size_average=True, reduction=True, epsilon=1e-8):
        """
        初始化 R2 类

        参数:
        - size_average (bool): 是否对误差进行平均处理
        - reduction (bool): 是否应用缩减操作（均值或求和）
        - epsilon (float): 小常数，用于避免总平方和为零的情况
        """
        super(R2, self).__init__()
        self.epsilon = epsilon            # 防止除以零的小常数
        self.size_average = size_average   # 是否对误差进行平均处理
        self.reduction = reduction        # 是否应用缩减操作（均值或求和）

    def __call__(self, predicted, actual):
        """
        计算决定系数 R^2

        参数:
        - predicted (torch.Tensor): 预测值，形状为 [batch_size, length]
        - actual (torch.Tensor): 实际值，形状为 [batch_size, length]

        返回:
        - r2 (torch.Tensor): 决定系数 R^2
        """
        # 确保 predicted 和 actual 是相同形状的张量
        assert predicted.shape == actual.shape, "预测值和实际值的形状必须相同"

        # 计算实际值的平均值（对每个样本单独计算均值）
        y_mean = torch.mean(actual, dim=1, keepdim=True)  # 形状: [batch_size, 1]

        # 计算残差平方和 SS_res（对每个样本单独计算）
        ss_res = torch.sum((actual - predicted) ** 2, dim=1)  # 形状: [batch_size]

        # 计算总平方和 SS_tot（对每个样本单独计算），添加 epsilon 以防 SS_tot 为零
        ss_tot = torch.sum((actual - y_mean) ** 2, dim=1) + self.epsilon  # 形状: [batch_size]

        # 计算 R^2（对每个样本单独计算）
        r2 = 1 - (ss_res / ss_tot)  # 形状: [batch_size]

        # 根据 reduction 参数决定是否进行缩减
        if self.reduction:
            if self.size_average:
                return torch.mean(r2)  # 返回所有样本的 R^2 均值
            else:
                return torch.sum(r2)  # 返回所有样本的 R^2 总和
        else:
            return r2  # 返回每个样本的 R^2 值，形状: [batch_size]