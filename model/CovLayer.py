import torch
import torch.nn as nn
import numpy as np


class CovLayer(nn.Module):
    def __init__(self, num_end, length, width):
        super(CovLayer, self).__init__()
        self.length, self.width = length, width
        self.cov = nn.Conv2d(num_end,
                             num_end,
                             kernel_size=(3, 3),
                             padding=int(np.floor(3 / 2)),
                             padding_mode='reflect',
                             bias=False,
                             groups=num_end)
        self.cov.load_state_dict({
            "weight":
            self.gauss_init(num_end, kernel_size=[3, 3], sigma=1e-3)
        })

    def forward(self, x):
        x1 = x.view(x.shape[0], x.shape[1], self.length, self.width)
        c = self.cov(x1)
        c = c.view(x.size())
        return c

    def gauss(self, x, y, sigma):
        z = 2 * np.pi * sigma**2
        return 1 / z * np.exp((-(x**2 + y**2) / 2 / sigma**2))

    def gauss_init(self,
                   num_end,
                   kernel_size=[3, 3],
                   sigma=1e-3):  # sigma = 1e-3
        # kernel_shape 是一个二元元组，各个元素分别表示：滤波器的宽，滤波器的高
        kernel = np.zeros(kernel_size, np.float32)
        mid = np.floor(kernel_size[0] / 2)
        for i in range(kernel_size[0]):
            for j in range(kernel_size[1]):
                kernel[i, j] = self.gauss(i - mid, j - mid, sigma)
        # kernel /= (np.sum(kernel) * self.num_end)
        kernel /= np.sum(kernel)
        kernels = torch.zeros(num_end, 1, kernel_size[0], kernel_size[1])
        kernels[:, :] = torch.tensor(kernel)
        return kernels
