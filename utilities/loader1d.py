'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:07:20
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 18:35:06
'''


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from . import utilkit as kit

class Loader1D:
    def __init__(self, in_chan, out_chan, data_path, ntrain, nvalid, itplen=None, batch_size=None):
        # Load initial data
        self.x_data, self.y_data, self.freq_data = kit.load_training_data(in_chan, out_chan, data_path, freq=True)
        self.freq_data=self.freq_data.unsqueeze(1)

        # Check dimensions
        self.x_len = self.x_data['x1_data'].shape[1]
        self.y_len = self.y_data['y1_data'].shape[1]

        # Create grid
        grid_idx = np.linspace(0, 1, num=itplen, endpoint=True)
        self.grid = torch.from_numpy(grid_idx)
        self.grid = self.grid.to(dtype=torch.float)
        self.grid = self.grid.reshape(1, self.grid.shape[0], 1)

        # Interpolate data
        if itplen:
            self.x_data, self.y_data = kit.interpolate(self.x_data, self.y_data, itplen, interpolator=kit.linear)

        # Save parameters
        self.ntrain = ntrain
        self.nvalid = nvalid
        self.batch_size = batch_size

    def gen_tensor(self, reerror=True, grid=True):
        # Split data into training and validation
        x_train, x_valid, y_train, y_valid = {}, {}, {}, {}
        for key, data in self.x_data.items():
            if self.ntrain + self.nvalid > data.shape[0] and reerror:
                raise OverflowError("The sum of training and validation set lengths exceeds the total data length.")
            x_train[key] = self.x_data[key][:self.ntrain, :]
            x_valid[key] = self.x_data[key][-self.nvalid:, :]

        for key, data in self.y_data.items():
            if self.ntrain + self.nvalid > data.shape[0] and reerror:
                raise OverflowError("The sum of training and validation set lengths exceeds the total data length.")
            y_train[key] = self.y_data[key][:self.ntrain, :]
            y_valid[key] = self.y_data[key][-self.nvalid:, :]
            
        freq_train = self.freq_data[:self.ntrain, :]
        freq_valid = self.freq_data[key][-self.nvalid:, :]  
        
        # Merge tensors
        x_train_merged = kit.merge_tensors(x_train)
        x_valid_merged = kit.merge_tensors(x_valid)
        y_train_merged = kit.merge_tensors(y_train)
        y_valid_merged = kit.merge_tensors(y_valid)
        
        # Merge grid
        if grid:
            grid = self.grid.repeat([x_train_merged.shape[0], 1, 1])
            x_train_merged = torch.cat((x_train_merged, grid), dim=-1)
            grid = self.grid.repeat([x_valid_merged.shape[0], 1, 1])
            x_valid_merged = torch.cat((x_valid_merged, grid), dim=-1)
            zeros = torch.zeros(freq_train.shape[0],freq_train.shape[1], 1)
            freq_train = torch.cat((freq_train, zeros), dim=-1)
            zeros = torch.zeros(freq_valid.shape[0],freq_valid.shape[1], 1)
            freq_valid = torch.cat((freq_valid, zeros), dim=-1)
       
        # Swap dimension (batch, in_channels, mode_in_dim)
        x_train_merged = x_train_merged.permute(0, 2, 1)
        x_valid_merged = x_valid_merged.permute(0, 2, 1)
        y_train_merged = y_train_merged.permute(0, 2, 1)
        y_valid_merged = y_valid_merged.permute(0, 2, 1)
        freq_train = freq_train.permute(0, 2, 1)
        freq_valid = freq_valid.permute(0, 2, 1)    
        
        return x_train_merged, x_valid_merged, y_train_merged, y_valid_merged, freq_train, freq_valid
        
    def gen_loaders(self, reerror=True, grid=True):
        # 将 x_data 和 y_data 分别按照键划分为训练和验证集
        x_train, x_valid, y_train, y_valid = {}, {}, {}, {}
        for key, data in self.x_data.items():
            if self.ntrain + self.nvalid > data.shape[0] and reerror:
                raise OverflowError("The sum of training and validation set lengths exceeds the total data length.")
            x_train[key] = self.x_data[key][:self.ntrain, :]
            x_valid[key] = self.x_data[key][-self.nvalid:, :]
        for key, data in self.y_data.items():
            if self.ntrain + self.nvalid > data.shape[0] and reerror:
                raise OverflowError("The sum of training and validation set lengths exceeds the total data length.")
            y_train[key] = self.y_data[key][:self.ntrain, :]
            y_valid[key] = self.y_data[key][-self.nvalid:, :]
            
        # 对频率数据直接截取训练与验证部分
        freq_train = self.freq_data[:self.ntrain, :]
        freq_valid = self.freq_data[-self.nvalid:, :]
        
        # 合并字典中的数据，使用 kit.merge_tensors
        x_train_merged = kit.merge_tensors(x_train)
        x_valid_merged = kit.merge_tensors(x_valid)
        y_train_merged = kit.merge_tensors(y_train)
        y_valid_merged = kit.merge_tensors(y_valid)
        
        # 如果要求添加 grid，则对 x 和频率数据进行额外拼接处理
        if grid:
            # 对 x 数据拼接 grid
            grid_train = self.grid.repeat([x_train_merged.shape[0], 1, 1])
            x_train_merged = torch.cat((x_train_merged, grid_train), dim=-1)
            grid_valid = self.grid.repeat([x_valid_merged.shape[0], 1, 1])
            x_valid_merged = torch.cat((x_valid_merged, grid_valid), dim=-1)
            
            # 对频率数据在最后一维添加零向量，使得其维度从 1 提升到 1+1=2
            zeros_train = torch.zeros(freq_train.shape[0], freq_train.shape[1], 1)
            freq_train = torch.cat((freq_train, zeros_train), dim=-1)
            zeros_valid = torch.zeros(freq_valid.shape[0], freq_valid.shape[1], 1)
            freq_valid = torch.cat((freq_valid, zeros_valid), dim=-1)
            
        # Swap dimension (batch, in_channels, mode_in_dim)
        x_train_merged = x_train_merged.permute(0, 2, 1)
        x_valid_merged = x_valid_merged.permute(0, 2, 1)
        y_train_merged = y_train_merged.permute(0, 2, 1)
        y_valid_merged = y_valid_merged.permute(0, 2, 1)
        freq_train = freq_train.permute(0, 2, 1)
        freq_valid = freq_valid.permute(0, 2, 1)   
        
        # 构造包含 (x, y, frequency) 三个输入的 TensorDataset
        train_dataset = TensorDataset(x_train_merged, freq_train, y_train_merged)
        valid_dataset = TensorDataset(x_valid_merged, freq_valid, y_valid_merged)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, valid_loader

    