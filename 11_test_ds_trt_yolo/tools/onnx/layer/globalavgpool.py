import torch.nn as nn
from torch.nn.functional import avg_pool2d

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x
