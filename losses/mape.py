"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:08:15
"""

import torch


class MAPE:
    """Mean absolute percentage error (MAPE)."""

    def __init__(self, size_average=True, reduction=True):
        self.size_average = size_average
        self.reduction = reduction

    def abs_percentage(self, x, y):
        num_examples = x.size(0)
        diff = x.reshape(num_examples, -1) - y.reshape(num_examples, -1)
        denom = y.reshape(num_examples, -1) + 1e-3
        abs_percentage_diff = torch.abs(diff / denom)
        all_mape = torch.mean(abs_percentage_diff, dim=1) * 100

        if self.reduction:
            return torch.mean(all_mape) if self.size_average else torch.sum(all_mape)
        return all_mape

    def __call__(self, x, y):
        return self.abs_percentage(x, y)
