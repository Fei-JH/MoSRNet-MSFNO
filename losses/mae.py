'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:07:58
'''

import torch

class MAE(object):
    def __init__(self, size_average=True, reduction=True):
        super(MAE, self).__init__()
        self.size_average = size_average
        self.reduction = reduction

    def abs(self, x, y):
        num_examples = x.size()[0]
        abs_diff = torch.abs(x.reshape(num_examples, -1) - y.reshape(num_examples, -1))
        all_mae = torch.mean(abs_diff, dim=1)  # the MAE for each example

        if self.reduction:
            if self.size_average:
                return torch.mean(all_mae)  # return the average MAE over all samples
            else:
                return torch.sum(all_mae)   # return the sum of MAE over all samples
        return all_mae

    def __call__(self, x, y):
        return self.abs(x, y)