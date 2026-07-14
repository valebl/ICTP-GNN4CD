import torch
import torch.nn as nn
from .registry import register_loss


@register_loss("Bernoulli_Gamma_NLL_Loss")
class Bernoulli_Gamma_NLL_Loss(nn.Module):
    """
    Bernoulli-Gamma (hurdle) negative log-likelihood for precipitation.
    """

    output_dim = 3
    components = ["p", "shape", "scale"]

    @staticmethod
    def add_loss_specific_args(parser):
        parser.add_argument("--threshold_nll", type=float)
        parser.add_argument("--magnitude_weight", type=float)
        return parser

    def __init__(self, ignore_nans: bool = True, eps: float = 1e-6,
                 threshold_nll: float | None = 0.1, magnitude_weight: float = 0.05):
        super().__init__()
        self.ignore_nans = ignore_nans
        self.eps = eps
        self.threshold_nll = threshold_nll
        self.magnitude_weight = magnitude_weight

    def forward(self, y_out, target):
        p_raw, shape_raw, scale_raw = y_out[:, 0], y_out[:, 1], y_out[:, 2]

        p = torch.sigmoid(p_raw)
        shape = torch.exp(shape_raw)
        scale = torch.exp(scale_raw)

        if self.threshold_nll is not None:
            target = target - self.threshold_nll
            target = target.clamp(min=0)

        if self.ignore_nans:
            mask = ~torch.isnan(target)
            target, p, shape, scale = target[mask], p[mask], shape[mask], scale[mask]

        rain = (target > 0).float()

        no_rain_ll = (1 - rain) * torch.log(1 - p + self.eps)
        rain_ll = rain * (
            torch.log(p + self.eps)
            + (shape - 1) * torch.log(target + self.eps)
            - shape * torch.log(scale + self.eps)
            - torch.lgamma(shape + self.eps)
            - target / (scale + self.eps)
        )
        nll = -(no_rain_ll + rain_ll)

        if self.magnitude_weight > 0:
            # note: target here is the shifted excess, so weighting is
            # relative to excess-over-threshold, not raw precipitation
            weight = (1.0 + self.magnitude_weight * target).detach()
            return (nll * weight).sum() / weight.sum()
        
        return nll.mean(), [p.mean(), shape.mean(), scale.mean()]
