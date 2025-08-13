# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:40:58 2024

@author: pluto
"""

import torch

class MAE(object):
    def __init__(self, size_average=True, reduction=True):
        super(MAE, self).__init__()
        self.size_average = size_average
        self.reduction = reduction

    def abs(self, x, y):
        num_examples = x.size()[0]
        abs_diff = torch.abs(x.reshape(num_examples, -1) - y.reshape(num_examples, -1))
        all_mae = torch.mean(abs_diff, dim=1)  # 每个样本的平均绝对误差

        if self.reduction:
            if self.size_average:
                return torch.mean(all_mae)  # 返回所有样本的平均MAE
            else:
                return torch.sum(all_mae)   # 返回所有样本的总和
        return all_mae

    def __call__(self, x, y):
        return self.abs(x, y)