'''
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 17:27:14
FilePath: \MS-FNO&MoSRNet_clean\models\common_modules.py
'''



import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        # out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FourierLayer(nn.Module):
    def __init__(self, modes, width):
        super(FourierLayer, self).__init__()
        self.modes1 = modes
        self.width = width


        self.padding = 2  # pad the domain if input is non-periodic

        # Create lists to hold the spectral conv and weight layers
        self.convs = SpectralConv1d(self.width, self.width, self.modes1)
        self.ws = nn.Conv1d(self.width, self.width, 1)
        
    def forward(self, x):
  
        # x = x.permute(0, 2, 1)  # Change the order to (batchsize, channels, sequence)

        x1 = self.convs(x)
        x2 = self.ws(x)
        x = x1 + x2
        x = F.gelu(x)  # Apply GELU after each layer's sum

        # x = x.permute(0, 2, 1)  # Change the order back to (batchsize, sequence, channels)
        return x
  
    



