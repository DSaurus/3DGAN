import torch
import torch.nn as nn
from utils.network_utils import IdentityBlock, ConvBlock, MLP

class GeneratorSDF(nn.Module):
    def __init__(self):
        super(GeneratorSDF, self).__init__()
        self.mlp = MLP([64+3, 128, 64, 32, 1], nn.Sigmoid())
    def forward(self, x):
        lines = torch.linspace(-1, 1, 128, dtype=x.dtype, device=x.device)
        a, b, c = torch.meshgrid(lines, lines, lines)
        pts = torch.cat([a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0)], dim=0)
        pts = pts.reshape(1, 3, 128, 128, 128).repeat(x.shape[0], 1, 1, 1, 1)
        feature = x.unsqueeze(2).repeat(1, 1, 128, 128, 128)
        feature = torch.cat([feature, pts], dim=1).reshape(x.shape[0], 64+3, -1)
        sdf = self.mlp(feature)
        sdf = sdf.reshape(x.shape[0], 128, 128, 128)
        return sdf

class Generator2D(nn.Module):
    def __init__(self):
        super(Generator2D, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.stage(x)
        return x
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv3d(160, 160, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(160, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(128, 96, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(96),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(96, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(32, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(16, 8, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(8),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(8, 1, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.stage(x)
        return x