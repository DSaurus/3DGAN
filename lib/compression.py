import torch.nn as nn

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