import torch
import torch.nn as nn
import torch.nn.functional as F
from model.OneCycle import oneCycle


class IVIUNet(nn.Module):
    def __init__(self, library, length, width, admm_layers):
        super(IVIUNet, self).__init__()
        num_bands, num_end = library.size()
        self.admm_layers = admm_layers
        self.oneCycle = oneCycle(library, num_end, length, width)
        self.recovery = nn.Linear(num_end, num_bands, bias=False)
        self.recovery.weight = nn.Parameter(library, False)
        # self.register_buffer('library', library)

    def forward(self, y):
        z_n, u_n = 0, 0
        for i in range(self.admm_layers):
            z_n, u_n = self.oneCycle(y, z_n, u_n)

        est_abu = F.normalize(z_n, p=1, dim=1)
        # rec_curve = self.library @ est_abu
        rec_curve = torch.transpose(self.recovery(torch.transpose(est_abu, 1, 2)), 1, 2)

        return est_abu, rec_curve
