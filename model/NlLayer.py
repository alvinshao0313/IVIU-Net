import torch
import torch.nn as nn
import torch.nn.functional as F


class NlLayer(nn.Module):
    def __init__(self):
        super(NlLayer, self).__init__()
        self.cita = nn.Parameter(torch.tensor(0.), True)

    def forward(self, c, u):
        z = F.relu(c - u - self.cita)

        return z
