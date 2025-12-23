"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:10:24
"""

import torch


class R2:
    """Coefficient of determination (R^2)."""

    def __init__(self, size_average=True, reduction=True, epsilon=1e-8):
        self.epsilon = epsilon
        self.size_average = size_average
        self.reduction = reduction

    def __call__(self, predicted, actual):
        assert predicted.shape == actual.shape, "Predicted and actual tensors must have the same shape."

        y_mean = torch.mean(actual, dim=1, keepdim=True)
        ss_res = torch.sum((actual - predicted) ** 2, dim=1)
        ss_tot = torch.sum((actual - y_mean) ** 2, dim=1) + self.epsilon

        r2 = 1 - (ss_res / ss_tot)

        if self.reduction:
            return torch.mean(r2) if self.size_average else torch.sum(r2)
        return r2
