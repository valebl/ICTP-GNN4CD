import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_loss


@register_loss("Bernoulli_Gamma_NLL_Loss")
class Bernoulli_Gamma_NLL_Loss(nn.Module):
    output_dim = 3 # class attribute

    def __init__(self, ignore_nans=True, eps=1e-6):
        super().__init__()
        self.ignore_nans = ignore_nans
        self.eps = eps

    def forward(self, y_out, target):
        # unpack raw parameters
        p_raw = y_out[:, 0]
        shape_raw = y_out[:, 1]
        scale_raw = y_out[:, 2]

        # transforms
        p = torch.sigmoid(p_raw)
        shape = F.softplus(shape_raw) + self.eps
        scale = F.softplus(scale_raw) + self.eps

        if self.ignore_nans:
            mask = ~torch.isnan(target)
            target = target[mask]
            p = p[mask]
            shape = shape[mask]
            scale = scale[mask]

        rain = (target > 0).float()

        # log-likelihood components
        no_rain_ll = (1 - rain) * torch.log(1 - p + self.eps)

        rain_ll = rain * (
            torch.log(p + self.eps)
            + (shape - 1) * torch.log(target + self.eps)
            - shape * torch.log(scale)
            - torch.lgamma(shape)
            - target / scale
        )

        return -(no_rain_ll + rain_ll).mean()


