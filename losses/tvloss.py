'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:11:09
'''

import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):

        super(TVLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, x):
        # calculate total variation loss
        diff = torch.abs(x[:, 1:] - x[:, :-1])
        if self.reduction == 'mean':
            # calculate mean TV loss
            tv_loss = self.weight * torch.mean(diff)
        elif self.reduction == 'sum':
            # calculate sum TV loss
            tv_loss = self.weight * torch.sum(diff)
        else:
            # no reduction
            tv_loss = self.weight * diff
        return tv_loss