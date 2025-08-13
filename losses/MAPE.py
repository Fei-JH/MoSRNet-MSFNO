'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 18:42:47
'''


import torch

class MAPE(object):
    def __init__(self, size_average=True, reduction=True):
        super(MAPE, self).__init__()
        self.size_average = size_average
        self.reduction = reduction

    def abs_percentage(self, x, y):
        num_examples = x.size()[0]
        abs_percentage_diff = torch.abs((x.reshape(num_examples, -1) - y.reshape(num_examples, -1)) / (y.reshape(num_examples, -1) + 1e-3))
        all_mape = torch.mean(abs_percentage_diff, dim=1) * 100  # 计算百分比误差并转为百分比

        if self.reduction:
            if self.size_average:
                return torch.mean(all_mape)  # 返回所有样本的平均MAPE
            else:
                return torch.sum(all_mape)   # 返回所有样本的总和
        return all_mape

    def __call__(self, x, y):
        return self.abs_percentage(x, y)