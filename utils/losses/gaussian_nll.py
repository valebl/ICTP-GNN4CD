import torch
import torch.nn as nn
from .registry import register_loss


@register_loss("Gaussian_NLL_Loss")
class Gaussian_NLL_Loss(nn.Module):
    output_dim = 2 # class attribute

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_out, target):
        mu = y_out[:, 0]
        sigma_raw = y_out[:, 1]

        # enforce positivity
        sigma = torch.nn.functional.softplus(sigma_raw) + self.eps

        # Gaussian negative log-likelihood
        nll = 0.5 * ((target - mu)**2 / (sigma**2) + 2 * torch.log(sigma))

        return nll.mean()




