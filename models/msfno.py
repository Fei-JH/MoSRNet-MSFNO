# ------------------------------------------------------------------------------------
# This script includes code adapted from the Neural Operator project:
# https://github.com/neuraloperator/neuraloperator
#
# MIT License

# Copyright (c) 2023 NeuralOperator developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. Performs FFT, linear transform, and inverse FFT.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes1 (int): Number of Fourier modes to multiply, at most floor(N/2) + 1.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        """
        Performs complex multiplication.
        Args:
            input (Tensor): Input tensor of shape (batch, in_channel, x).
            weights (Tensor): Weight tensor of shape (in_channel, out_channel, x).
        Returns:
            Tensor: Output tensor of shape (batch, out_channel, x).
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Forward pass of SpectralConv1d.
        Args:
            x (Tensor): Input tensor of shape (batch, in_channels, length).
        Returns:
            Tensor: Output tensor of shape (batch, out_channels, length).
        """
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FourierLayer(nn.Module):
    def __init__(self, modes, width):
        super(FourierLayer, self).__init__()
        """
        Fourier layer with spectral convolution and pointwise convolution.
        Args:
            modes (int): Number of Fourier modes.
            width (int): Channel width.
        """
        self.modes1 = modes
        self.width = width
        self.padding = 2  # Pad the domain if input is non-periodic
        self.convs = SpectralConv1d(self.width, self.width, self.modes1)
        self.ws = nn.Conv1d(self.width, self.width, 1)
        
    def forward(self, x):
        """
        Forward pass of FourierLayer.
        Args:
            x (Tensor): Input tensor of shape (batch, width, length).
        Returns:
            Tensor: Output tensor of shape (batch, width, length).
        """
        x1 = self.convs(x)
        x2 = self.ws(x)
        x = x1 + x2
        x = F.gelu(x)
        return x

class MSFNO(nn.Module):
    def __init__(self, in_channels, mode_length, embed_dim, fno_modes, fno_layers, out_channels):
        """
        Modal-stiffness Fourier Neural Operator (MS-FNO).
        Args:
            in_channels (int): Number of input tokens.
            mode_length (int): Length of modal input (including positional encoding); also target dimension after frequency lifting.
            embed_dim (int): Embedding dimension.
            fno_modes (int): Number of Fourier modes in FNO layers.
            fno_layers (int): Number of FNO layers.
            out_channels (int): Output dimension.
        """
        super(MSFNO, self).__init__()
        self.lifting = nn.Linear(in_channels, embed_dim)
        self.fno_layers = nn.ModuleList([
            FourierLayer(fno_modes, embed_dim) for _ in range(fno_layers)
        ])
        self.projection1 = nn.Linear(embed_dim, embed_dim // 2)
        self.projection2 = nn.Linear(embed_dim // 2, out_channels)

    def forward(self, x):
        """
        Forward pass of MSFNO.
        Args:
            x (Tensor): Input tensor of shape (batch, in_channels, mode_length).
        Returns:
            Tensor: Output tensor of shape (batch, mode_length, out_channels).
        """
        x = self.lifting(x.transpose(1, 2)).transpose(1, 2)  # (batch, embed_dim, mode_length)
        for layer in self.fno_layers:
            x = F.gelu(layer(x))
        x = x.transpose(1, 2)
        x = self.projection1(x)
        out = self.projection2(x)
        return out
