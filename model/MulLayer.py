import torch
import torch.nn as nn


class MulLayer(nn.Module):
    def __init__(self):
        super(MulLayer, self).__init__()
        self.eta = nn.Parameter(torch.tensor(1e-2), True)

    def forward(self, u_k, c, z):
        u = u_k + self.eta * (c - z)
        return u
