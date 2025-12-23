'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:12:13
'''

import torch
import torch.nn as nn

    
class Subnetwork(nn.Module):
    def __init__(self, dim1, dim2fct, inlen, outlen):
        super().__init__()
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
        )

    def forward(self, x):
        return self.layers(x)


class MoSRNet(nn.Module):
    def __init__(
        self, dim1: int = 32, dim2fct: int = 2, inlen: int = 5, outlen: int = 541, num_subnets: int = 3
    ):
        super().__init__()
        self.subnets = nn.ModuleList([
            Subnetwork(dim1=dim1, dim2fct=dim2fct, inlen=inlen, outlen=outlen)
            for _ in range(num_subnets)
        ])

    def forward(self, x):
        # x shape: (batch, num_subnets, length)
        outputs = []
        for i, subnet in enumerate(self.subnets):
            out = subnet(x[:, i : i + 1, :])  # (batch, 1, length)
            outputs.append(out.squeeze(1))  # (B, L)
        return torch.stack(outputs, dim=1)  # (batch, num_subnets, length)
