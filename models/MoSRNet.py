# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 15:36:54 2025 (JST)

@author: Jinghao FEI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
                      
    
class Subnetwork(nn.Module):
    def __init__(self, dim1, dim2fct, inlen, outlen):
        super(Subnetwork, self).__init__()
        self.dim1 = dim1
        self.dim2fct = dim2fct
        self.inlen = inlen
        self.outlen = outlen
        self.dim2 = self.dim1 * self.dim2fct
        
        self.layers = nn.Sequential(
            nn.Conv1d(1, self.dim1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.dim1, self.dim2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(self.dim2),
            nn.Flatten(),
            nn.Linear(self.dim2 * self.inlen, self.outlen)  

    def forward(self, x):
        x = self.layers(x)
        return x

# === Wrapper network remains unchanged, just swap sub-net ===
class MoSRNet(nn.Module):
    def __init__(self, dim1: int = 32, dim2fct: int = 2, inlen:int = 5, outlen: int = 541, num_subnets: int = 3):
        super().__init__()
        self.subnets = nn.ModuleList([
            Subnetwork(dim1=dim1, dim2fct=dim2fct, inlen=inlen, outlen=outlen)
            for _ in range(num_subnets)
        ])

    def forward(self, x):
        # x shape : (B, num_subnets, L)
        outputs = []
        for i, sulnet in enumerate(self.subnets):
            out = sulnet(x[:, i:i+1, :])  # (B, 1, L)
            outputs.append(out.squeeze(1))  # (B, L)
        return torch.stack(outputs, dim=1)  # (B, num_subnets, L)
