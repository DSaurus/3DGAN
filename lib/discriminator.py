import torch
import torch.nn as nn
from utils.network_utils import IdentityBlock, ConvBlock

class Discriminator2D(nn.Module):
    def __init__(self):
        super(Discriminator2D, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),   
        )
        self.last_stage = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.stage(x)
        y = self.last_stage(x)
        return y, x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 96, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 160, 3, stride=2, padding=1),   
        )
        self.last_stage = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(160, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.stage(x)
        y = self.last_stage(x)
        return y, x