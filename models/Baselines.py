
import torch
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
        参数说明:
          in_channels: 输入通道数
          mode_length: 模型 mode 的长度
          freq_length: 频率信息维度
          embed_dim: 嵌入维度
          fno_modes: 保留，与原模型一致
          fno_layers: 不再用于残差模块数，使用 ResNet18-1D 规范 [2,2,2,2]
          out_channels: 输出通道数
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

        # ResNet18-1D 风格：4 个阶段，每阶段 2 个 BasicBlock1D
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

        # Layer normalization 在输出 projection 之前
        self.norm1 = nn.LayerNorm(embed_dim)
        # 输出层：两层全连接网络
        self.projection1 = nn.Linear(embed_dim, embed_dim // 2)
        self.projection2 = nn.Linear(embed_dim // 2, out_channels)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        # 第一个 block
        layers.append(BasicBlock1D(in_ch, out_ch, stride))
        # 后续 blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        输入:
          mode_shapes: (batch, in_channels, mode_length)
          frequencies: (batch, in_channels, freq_length)
        输出:
          out: (batch, mode_length, out_channels)
        """
        # Embedding & 初步处理
        x = F.gelu(self.mode_embed(x.transpose(1, 2))).transpose(1, 2)

        # 初始大卷积
        x = self.initial_conv(x)
        # ResNet18-1D 风格特征提取
        x = self.resnet_layers(x)

        # 准备 LayerNorm，shape 转为 (batch, mode_length, embed_dim)
        x = x.transpose(1, 2)
        x = self.norm1(x)

        # 输出预测
        x = self.projection1(x)
        out = self.projection2(x)
        return out


    

class DNN(nn.Module):
    def __init__(self, in_channels, mode_length, embed_dim, fno_modes, fno_layers, out_channels):
        """
        参数说明:
          in_channels: 输入通道数
          mode_length: 模型mode的长度
          freq_length: 频率信息维度
          embed_dim: 嵌入维度
          fno_modes: （保留，与原模型一致）
          fno_layers: 作为全连接层的层数（可以调深）
          out_channels: 输出通道数
        """
        super().__init__()
        # Embedding layers
        self.mode_embed = nn.Linear(in_channels, embed_dim)
        
        # DNN部分：对每个时间步独立应用多层全连接网络
        layers = []
        for _ in range(5):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(nn.ReLU())
        self.dnn_layers = nn.Sequential(*layers)

        # Layer normalization (在全连接网络之后对每个时间步进行归一化)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 输出层
        self.projection1 = nn.Linear(embed_dim, embed_dim // 2)
        self.projection2 = nn.Linear(embed_dim // 2, out_channels)

    def forward(self, x):
        """
        输入:
          mode_shapes: (batch, in_channels, mode_length)
          frequencies: (batch, in_channels, freq_length)
        输出:
          out: (batch, mode_length, out_channels)
        """
        # Embedding & 融合
        x = F.gelu(self.mode_embed(x.transpose(1, 2))).transpose(1, 2)

        # 将fused调整为 (batch, mode_length, embed_dim) 以应用DNN（对每个时间步独立处理）
        x = x.transpose(1, 2)
        x = self.dnn_layers(x)
        
        # Layer normalization
        x = self.norm1(x)
        
        # 输出层
        x = self.projection1(x)
        out = self.projection2(x)
        return out
    
