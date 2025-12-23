"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:11:09
"""

import torch
import torch.nn as nn


class TVLoss(nn.Module):
    """Total variation loss."""

    def __init__(self, weight=1.0, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, x):
        diff = torch.abs(x[:, 1:] - x[:, :-1])
        if self.reduction == "mean":
            tv_loss = self.weight * torch.mean(diff)
        elif self.reduction == "sum":
            tv_loss = self.weight * torch.sum(diff)
        else:
            tv_loss = self.weight * diff
        return tv_loss
