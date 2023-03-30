import torch
import torch.nn as nn


class RecLayer(nn.Module):
    def __init__(self, library):
        super(RecLayer, self).__init__()
        self.register_buffer('libraryT', library.t())
        self.register_buffer('ll', library.t() @ library)
        self.H = nn.Parameter(torch.diag(torch.ones(library.size(1))))
        self.rou = nn.Parameter(torch.tensor(1e-2))

    def forward(self, y, z, u):
        x = torch.linalg.inv(self.ll + self.H) @ (self.libraryT @ y + self.rou * (z - u))

        return x
