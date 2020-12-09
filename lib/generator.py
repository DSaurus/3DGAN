import torch
import torch.nn as nn
from utils.network_utils import IdentityBlock, ConvBlock

class Discriminator(nn.Module):
    def __init__(self):
        super(ResNet3d, self).__init__()
        self.stage = nn.Sequential(
            IdentityBlock(512, [128, 128, 256]),
            nn.Upsample(scale_factor=2),
            IdentityBlock(256, [64, 64, 128]),
            nn.Upsample(scale_factor=2),
            IdentityBlock(128, [32, 32, 64]),
            nn.Upsample(scale_factor=2),
            IdentityBlock(64, [16, 16, 32]),
            nn.Upsample(scale_factor=2),
            IdentityBlock(32, [8, 8, 16]),
            nn.Upsample(scale_factor=2),
            IdentityBlock(32, [8, 8, 16]),
            nn.Upsample(scale_factor=2),
            IdentityBlock(16, [8, 8, 16]),
            nn.Upsample(scale_factor=2),
            IdentityBlock(16, [8, 8, 1]),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.stage(x)
        return x