'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 18:43:37
'''

import torch

class RMSE(object):
    def __init__(self, size_average=True, reduction=True):
        super(RMSE, self).__init__()
        self.size_average = size_average
        self.reduction = reduction

    def loss(self, x, y):
        # Number of examples in the batch
        num_examples = x.size()[0]
        # Reshape inputs to (num_examples, -1) and compute squared differences
        squared_diff = (x.reshape(num_examples, -1) - y.reshape(num_examples, -1)) ** 2
        # Calculate mean squared error for each sample
        all_mse = torch.mean(squared_diff, dim=1)
        # Compute square root to obtain RMSE for each sample
        all_rmse = torch.sqrt(all_mse)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_rmse)  # Return the average RMSE across all samples
            else:
                return torch.sum(all_rmse)   # Return the sum of RMSE across all samples
        return all_rmse

    def __call__(self, x, y):
        return self.loss(x, y)
