'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 17:26:47
FilePath: \MS-FNO&MoSRNet_clean\models\MSFNO.py
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common_modules import FourierLayer

# MS-FNO: Modal-stiffness Fourier Neural Operator
class MSFNO(nn.Module):
    def __init__(self, in_channels, mode_length,  embed_dim, fno_modes, fno_layers, out_channels):
        """
        MS-FNO (Modal-stiffness Fourier Neural Operator)
        参数:
          in_channels: 输入中 token 数（原来的 m）
          mode_length: 振型输入最后一维（包含位置编码）；同时作为频率提升后的目标维度（默认为16）
          embed_dim: 嵌入后维度（默认128）
          fno_modes: FNO层傅里叶模数（默认4）
          fno_layers: FNO层数（默认3）
          out_channels: 输出维度（默认1）
        """
        super(MSFNO, self).__init__()
        # 嵌入层沿 token 方向（in_channels → embed_dim）
        self.lifting = nn.Linear(in_channels, embed_dim)
        # 先将频率从1提升到 mode_length
        
        self.fno_layers = nn.ModuleList([
            FourierLayer(fno_modes, embed_dim) for _ in range(fno_layers)
        ])
        
        self.projection1 = nn.Linear(embed_dim, embed_dim//2)
        self.projection2 = nn.Linear(embed_dim//2, out_channels)

    def forward(self, x):
        """
        x: (batch, in_channels, mode_length)
        """
        # 对振型：先转置，将 token 数 in_channels 放到最后，再嵌入
        x = self.lifting(x.transpose(1, 2)).transpose(1, 2)  # (batch, embed_dim, mode_length)

        # FNO层处理
        for layer in self.fno_layers:
            x = F.gelu(layer(x))
        # 转置回 (batch, mode_length, embed_dim)
        x = x.transpose(1,2)
        x = self.projection1(x)
        out = self.projection2(x)  # (batch, mode_length, out_channels)
        return out

