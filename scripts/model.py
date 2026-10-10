import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_filters=16):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, 2 * f)
        self.enc3 = ConvBlock(2 * f, 4 * f)
        self.bottleneck = ConvBlock(4 * f, 8 * f)
        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(8 * f, 4 * f, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(8 * f, 4 * f)
        self.up2 = nn.ConvTranspose3d(4 * f, 2 * f, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(4 * f, 2 * f)
        self.up1 = nn.ConvTranspose3d(2 * f, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(2 * f, f)
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
