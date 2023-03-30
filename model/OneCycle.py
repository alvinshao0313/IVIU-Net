import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MulLayer import MulLayer
from model.NlLayer import NlLayer
from model.CovLayer import CovLayer
from model.RecLayer import RecLayer


class oneCycle(nn.Module):
    def __init__(self, library, num_end, length, width):
        super(oneCycle, self).__init__()
        self.recLayer = RecLayer(library)
        self.covLayer = CovLayer(num_end, length, width)
        self.nlLayer = NlLayer()
        self.mulLayer = MulLayer()

    def forward(self, y, z_k, u_k):
        x_k1 = self.recLayer(y, z_k, u_k)
        c_k1 = self.covLayer(x_k1)
        z_k1 = self.nlLayer(c_k1, u_k)
        u_k1 = self.mulLayer(u_k, c_k1, z_k1)
        return z_k1, u_k1
