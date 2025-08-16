import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelBranch(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=2):
        super().__init__()
        layers = []
        for _ in range(blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

b = ParallelBranch(3,32)
print('ParallelBranch ok. param count:', sum(p.numel() for p in b.parameters()))

a = torch.randn(1,3,256,256)
out = b(a)
print('forward output shape:', out.shape)
