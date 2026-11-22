import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoCnnLayers(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = TwoCnnLayers(in_channels, out_channels)
        self.maxPool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.block(x)
        y = self.maxPool(x)
        return y, x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convT = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.block = TwoCnnLayers(in_channels, out_channels)
    def forward(self, x, y):
        x = self.convT(x)
        cat = torch.cat([x, y], dim=1)
        y = self.block(cat)
        return y

class OutConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)