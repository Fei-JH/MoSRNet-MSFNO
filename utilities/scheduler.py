# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:59:24 2024

@author: pluto
"""
import math
import torch

class ExpLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, decay_rate, initial_ratio=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.initial_ratio = initial_ratio  # 初始学习率为最终学习率的比例
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]  # 读取 optimizer 初始学习率
        super(ExpLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：从 initial_ratio * base_lr 线性增长到 base_lr
            scale = self.initial_ratio + (1 - self.initial_ratio) * (self.last_epoch + 1) / self.warmup_epochs
        else:
            # 指数衰减阶段
            scale = self.decay_rate ** (self.last_epoch - self.warmup_epochs)

        return [base_lr * scale for base_lr in self.base_lrs]
    
    
class SinExpLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, decay_rate, initial_ratio=0, sin_amplitude=0.1, sin_frequency=0.1, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of warmup epochs.
            decay_rate (float): Exponential decay factor after warmup.
            sin_amplitude (float): Amplitude of the sinusoidal modulation.
            sin_frequency (float): Frequency of the sinusoidal modulation (cycles per epoch).
            initial_ratio (float): Starting learning rate ratio relative to base_lr during warmup.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.initial_ratio = initial_ratio
        self.sin_amplitude = sin_amplitude
        self.sin_frequency = sin_frequency
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super(SinExpLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase from initial_ratio * base_lr to base_lr.
            scale = self.initial_ratio + (1 - self.initial_ratio) * (self.last_epoch + 1) / self.warmup_epochs
            sin_factor = 1.0  # Do not add sinusoidal modulation during warmup.
        else:
            # Exponential decay phase.
            scale = self.decay_rate ** (self.last_epoch - self.warmup_epochs+1)
            # Add sinusoidal modulation.
            sin_factor = 1 + self.sin_amplitude * math.sin(2 * math.pi * self.sin_frequency * (self.last_epoch-self.warmup_epochs+1))
        return [base_lr * scale * sin_factor for base_lr in self.base_lrs]
    
    
class AdaptiveLRMomentumScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_factor=0.1, patience=10, min_lr=1e-6, last_epoch=-1):
        self.decay_factor = decay_factor
        self.patience = patience
        self.min_lr = min_lr
        self.num_bad_epochs = 0
        self.best_loss = float('inf')
        super(AdaptiveLRMomentumScheduler, self).__init__(optimizer, last_epoch)

    def step(self, metrics):
        if metrics < self.best_loss:
            self.best_loss = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group['lr'] * self.decay_factor, self.min_lr)
                param_group['lr'] = new_lr
            self.num_bad_epochs = 0  # Reset patience counter

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class PolynomialDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, power=2.0, min_lr=1e-6, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        decay_factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [max(base_lr * decay_factor, self.min_lr) for base_lr in self.base_lrs]
