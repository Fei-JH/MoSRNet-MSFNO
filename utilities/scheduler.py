# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:59:24 2024

@author: pluto
"""
import torch


class ExpLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Exponential decay with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, decay_rate, initial_ratio=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.initial_ratio = initial_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup: linearly increase from initial_ratio * base_lr to base_lr.
            scale = self.initial_ratio + (1 - self.initial_ratio) * (self.last_epoch + 1) / self.warmup_epochs
        else:
            # Exponential decay after warmup.
            scale = self.decay_rate ** (self.last_epoch - self.warmup_epochs)
        return [base_lr * scale for base_lr in self.base_lrs]
