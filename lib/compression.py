import torch.nn as nn
import torch
from utils.network_utils import MLP

class CompressionSDF(nn.Module):
    def __init__(self):
        super(CompressionSDF, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 1)
        )
        self.mlp = MLP([17, 32, 32, 16, 1], nn.Sigmoid())
    def forward(self, x):
        feature = self.stage(x)
        pts = torch.linspace(-1, 1, 128, dtype=feature.dtype, device=feature.device).reshape(1, 1, 128, 1, 1).repeat(x.shape[0], 1, 1, 128, 128)
        feature = feature.unsqueeze(2).repeat(1, 1, 128, 1, 1)
        feature = torch.cat([feature, pts], dim=1).reshape(x.shape[0], 16+1, -1)
        sdf = self.mlp(feature)
        sdf = sdf.reshape(x.shape[0], 128, 128, 128)
        return sdf

class Compression2D(nn.Module):
    def __init__(self):
        super(Compression2D, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 16, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.stage(x)