import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

class SEUnit(nn.Module):
    def __init__(self, channels):
        super(SEUnit, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvolutionalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, multi_scale=True):
        super(ConvolutionalUnit, self).__init__()
        self.multi_scale = multi_scale
        
        if multi_scale:
            self.conv3x3 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=3, padding=1)
            self.conv5x5 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=5, padding=2)
            self.conv7x7 = nn.Conv2d(in_channels, out_channels - 2*(out_channels // 3), kernel_size=7, padding=3)
        else:
            self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        
        self.se = SEUnit(out_channels)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.multi_scale:
            out = torch.cat([self.conv3x3(x), self.conv5x5(x), self.conv7x7(x)], dim=1)
        else:
            out = self.conv7x7(x)
        
        out = self.se(out)
        out = self.conv1x1(out)
        return out + x

class AlignmentModule(nn.Module):
    def __init__(self, in_channels, out_channels, N_A):
        super(AlignmentModule, self).__init__()
        
        self.conv5x5_1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        self.conv5x5_2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv1x1 = nn.Conv2d(64, 64, kernel_size=1)
        
        self.cu_layers = nn.Sequential(*[ConvolutionalUnit(64, 64) for _ in range(N_A)])
        
        self.upconv = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.final_conv1x1 = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv5x5_1(x)
        x = self.conv5x5_2(x1)
        x = self.conv1x1(x)
        x = self.cu_layers(x)
        x = self.upconv(x)
        x = torch.cat([x, x1], dim=1)
        x = self.final_conv1x1(x)
        return x

class EVRNet(nn.Module):
    def __init__(self, N_A=5):
        super(EVRNet, self).__init__()
        self.conv3x3_It = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv3x3_It_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.alignment = AlignmentModule(64 + 64 + 2, 64, N_A)
        
    def forward(self, I_t, I_t_1, H_t_1):
        I_t_conv = self.conv3x3_It(I_t)
        I_t_1_conv = self.conv3x3_It_1(I_t_1)
        
        x = torch.cat([I_t_conv, I_t_1_conv, H_t_1], dim=1)
        A_t = self.alignment(x)
        
        return A_t
