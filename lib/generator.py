import torch
import torch.nn as nn
from utils.network_utils import IdentityBlock, ConvBlock

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv3d(512, 160, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(160, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(128, 96, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(96),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(96, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(32, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(16, 8, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(8),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(8, 1, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.stage(x)
        return x