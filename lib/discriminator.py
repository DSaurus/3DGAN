import torch
import torch.nn as nn
from utils.network_utils import IdentityBlock, ConvBlock

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.stage = nn.Sequential(
            ConvBlock(1, [8, 8, 16]),
            ConvBlock(16, [16, 16, 32]),
            ConvBlock(32, [16, 16, 64]),
            ConvBlock(64, [32, 32, 128]),
            ConvBlock(128, [64, 64, 256]),
            ConvBlock(256, [64, 64, 256]),
            ConvBlock(256, [128, 128, 512]),
        )
        self.last_stage = nn.Sequential(
            IdentityBlock(512, [64, 64, 128]),
            IdentityBlock(128, [16, 16, 32]),
            IdentityBlock(32, [8, 8, 1]),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.stage(x)
        y = self.last_stage(x)
        return y