import torch.nn as nn
from torch import tanh
from torch.nn.functional import softplus

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (tanh(softplus(x)))
        return x
