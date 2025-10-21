'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:10:24
'''

import torch

class R2(object):
    def __init__(self, size_average=True, reduction=True, epsilon=1e-8):

        super(R2, self).__init__()
        self.epsilon = epsilon            
        self.size_average = size_average   
        self.reduction = reduction        

    def __call__(self, predicted, actual):
        
        assert predicted.shape == actual.shape, "Predicted and actual tensors must have the same shape."

        # calculate the mean of actual values for each sample
        y_mean = torch.mean(actual, dim=1, keepdim=True)  # [batch_size, 1]

        # calculate residual sum of squares SS_res (for each sample)
        ss_res = torch.sum((actual - predicted) ** 2, dim=1)  # [batch_size]

        # calculate total sum of squares SS_tot (for each sample)
        ss_tot = torch.sum((actual - y_mean) ** 2, dim=1) + self.epsilon  # [batch_size]

        # calculate R^2 for each sample
        r2 = 1 - (ss_res / ss_tot)  # [batch_size]

        if self.reduction:
            if self.size_average:
                return torch.mean(r2)  # return the average R^2 over all samples
            else:
                return torch.sum(r2)  # return the sum of R^2 over all samples
        else:
            return r2  