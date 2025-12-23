"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 18:44:46
"""

import torch


class RMSRE:
    """Root mean squared relative error (percentage)."""

    def __init__(self, size_average=True, reduction=True, epsilon=1e-3):
        self.size_average = size_average
        self.reduction = reduction
        self.epsilon = epsilon

    def __call__(self, predicted, actual):
        actual_safe = actual + self.epsilon * (actual == 0).float()

        relative_errors = torch.abs((predicted - actual) / actual_safe)
        squared_relative_errors = relative_errors ** 2

        if self.reduction:
            mse = torch.mean(squared_relative_errors) if self.size_average else torch.sum(squared_relative_errors)
        else:
            mse = squared_relative_errors

        rmsre = torch.sqrt(mse) * 100
        return rmsre
