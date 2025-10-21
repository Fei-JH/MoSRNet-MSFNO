'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-10-21 16:15:38
'''

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First 3x1 convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        # Second 3x1 convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, embed_dim, out_channels):
        """
        Parameter descriptions:
          in_channels: number of input channels
          mode_length: length of the model's mode dimension
          freq_length: dimensionality of the frequency information
          embed_dim: embedding dimension
          fno_modes: retained for compatibility with the original model
          fno_layers: no longer used to specify residual block counts; use ResNet18-1D configuration [2,2,2,2]
          out_channels: number of output channels
        """
        super().__init__()
        # Embedding layers
        self.mode_embed = nn.Linear(in_channels, embed_dim)

        # Initial large-kernel convolution (analogous to ResNet-18's 7x7 conv)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # ResNet18-1D layers
        self.layer1 = self._make_layer(embed_dim, embed_dim, blocks=2, stride=1)
        self.layer2 = self._make_layer(embed_dim, embed_dim, blocks=2, stride=1)
        self.layer3 = self._make_layer(embed_dim, embed_dim, blocks=2, stride=1)
        self.layer4 = self._make_layer(embed_dim, embed_dim, blocks=2, stride=1)
        self.resnet_layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

        # Layer normalization 
        self.norm1 = nn.LayerNorm(embed_dim)
        # output projection layers
        self.projection1 = nn.Linear(embed_dim, embed_dim // 2)
        self.projection2 = nn.Linear(embed_dim // 2, out_channels)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []

        layers.append(BasicBlock1D(in_ch, out_ch, stride))

        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        inputs:
          mode_shapes: (batch, in_channels, mode_length)
          frequencies: (batch, in_channels, freq_length)
        outputs[]:
          out: (batch, mode_length, out_channels)
        """
        # Embedding and transpose to (batch, embed_dim, mode_length)
        x = F.gelu(self.mode_embed(x.transpose(1, 2))).transpose(1, 2)

        # intial conv
        x = self.initial_conv(x)
        # ResNet18-1D 
        x = self.resnet_layers(x)

        # LayerNorm
        x = x.transpose(1, 2)
        x = self.norm1(x)

        # projection to output
        x = self.projection1(x)
        out = self.projection2(x)
        return out

