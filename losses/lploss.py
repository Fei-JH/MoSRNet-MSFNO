"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:08:41
"""

import operator
from functools import reduce

import torch


class LpLoss:
    """Lp loss with absolute and relative variants."""

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size(0)
        h = 1.0 / (x.size(1) - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )

        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size(0)

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        rel_norms = diff_norms / y_norms
        if self.reduction:
            return torch.mean(rel_norms) if self.size_average else torch.sum(rel_norms)
        return rel_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def count_params(model):
    """Count parameters, treating complex parameters as two real values."""
    total = 0
    for param in list(model.parameters()):
        size = list(param.size() + (2,) if param.is_complex() else param.size())
        total += reduce(operator.mul, size)
    return total
