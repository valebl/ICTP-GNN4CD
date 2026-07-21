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

    def forward(self, y_out, target, return_mean=False):
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

        loss = nll.mean()
        
        if return_mean:
            conditional_mean = shape * scale
            if self.threshold_nll is not None:
                conditional_mean = conditional_mean + self.threshold_nll
            mean = p * conditional_mean

            return loss, [p.mean(), shape.mean(), scale.mean(), mean]
        
        return loss, [p.mean(), shape.mean(), scale.mean()]
