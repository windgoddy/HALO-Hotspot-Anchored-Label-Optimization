import torch
import torch.nn as nn


class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        bottleneck_channels = int(channels // r)
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.Sigmoid(),
        )
        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
        )

    def forward(self, high, low):
        topdown_weight = self.topdown(high)
        bottomup_weight = self.bottomup(low)
        fused = 2 * torch.mul(low, topdown_weight) + 2 * torch.mul(high, bottomup_weight)
        return self.post(fused)
