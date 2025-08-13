# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:23:41 2024 (JST)

@author: Jinghao FEI
"""
import torch

class RMSRE(object):
    def __init__(self, size_average=True, reduction=True, epsilon=1e-3):
        super(RMSRE, self).__init__()
        self.size_average = size_average  # Whether to return the mean of the errors
        self.reduction = reduction  # Whether to apply reduction (mean or sum) to the errors
        self.epsilon = epsilon  # Small constant to prevent division by zero

    def __call__(self, predicted, actual):
        # Ensure that predicted and actual are tensors of the same shape
        num_examples = predicted.size()[0]
        
        # Avoid division by zero by adding a small constant epsilon where actual values are zero
        actual_safe = actual + self.epsilon * (actual == 0).float()

        # Calculate the relative error between each predicted and actual value
        relative_errors = torch.abs((predicted - actual) / actual_safe)
        
        # Compute the square of the relative errors
        squared_relative_errors = relative_errors ** 2
        
        # Sum or average the squared errors based on the reduction and size_average settings
        if self.reduction:
            if self.size_average:
                mse = torch.mean(squared_relative_errors)
            else:
                mse = torch.sum(squared_relative_errors)
        else:
            mse = squared_relative_errors

        # Calculate the root of the mean square error and convert to percentage
        rmsre = torch.sqrt(mse)* 100   # Convert to percentage

        return rmsre

