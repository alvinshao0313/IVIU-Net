import torch
import torch.nn as nn


class ModelLoss(nn.Module):
    def __init__(self, library, alpha1, alpha2, alpha3):
        super(ModelLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def forward(self, rec_curve, mea_curve):
        mse = torch.mean(torch.norm(rec_curve - mea_curve, p=2, dim=1))
        # clamp：限制范围在0~1，防止精度误差
        sad = torch.mean(
            torch.acos(
                torch.clamp(
                    ((torch.sum((rec_curve * mea_curve + 1e-5), dim=1)) /
                     (torch.norm(rec_curve, p=2, dim=1) *
                      torch.norm(mea_curve, p=2, dim=1) + 1e-5)), 0., 1.)))

        sid = torch.mean(
            torch.sum(
                rec_curve *
                (torch.log(torch.abs(rec_curve) + 1e-5) - torch.log(torch.abs(mea_curve) + 1e-5)) +
                mea_curve *
                (torch.log(torch.abs(mea_curve) + 1e-5) - torch.log(torch.abs(rec_curve) + 1e-5)),
                dim=1))

        return {
            'MSE': self.alpha1 * mse,
            'SAD': self.alpha2 * sad,
            'SID': self.alpha3 * sid,
        }
