"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 18:49:17
"""

import torch


class MSE:
    """Mean squared error (MSE)."""

    def __init__(self, size_average=True, reduction=True):
        self.size_average = size_average
        self.reduction = reduction

    def loss(self, x, y):
        num_examples = x.size(0)
        squared_diff = (x.reshape(num_examples, -1) - y.reshape(num_examples, -1)) ** 2
        all_mse = torch.mean(squared_diff, dim=1)

        if self.reduction:
            return torch.mean(all_mse) if self.size_average else torch.sum(all_mse)
        return all_mse

    def __call__(self, x, y):
        return self.loss(x, y)
