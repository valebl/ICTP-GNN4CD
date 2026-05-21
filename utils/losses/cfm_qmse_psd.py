import torch
import torch.nn as nn
import torch.nn.functional as F

from .psd import PSD_Loss
from .qmse import QMSE_Loss
from .registry import register_loss


@register_loss("CFM_QMSE_PSD_Loss")
class CFM_QMSE_PSD_Loss(nn.Module):
    output_dim = 1
    use_bins = True
    components = ["FM", "QMSE", "PSD"]

    @staticmethod
    def add_loss_specific_args(parser):
        parser.add_argument("--alpha", type=float)
        parser.add_argument("--beta", type=float)
        parser.add_argument("--binmin", type=float)
        parser.add_argument("--binmax", type=float)
        parser.add_argument("--binwidth", type=float)
        parser.add_argument("--binscale", type=str)
        return parser

    def __init__(self, alpha, beta, balance=None, *psd_args, **psd_kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.qmse_loss_fn = QMSE_Loss(balance)
        self.psd_loss_fn = PSD_Loss(apply_expm1=True, *psd_args, **psd_kwargs)

    def forward(self, out, target, bins):
        if type(out) == list:
            out = out[0]

        target = target.reshape(-1)
        v_pred = out[:, 0]
        v_target = out[:, 1]

        loss_fm = F.mse_loss(v_pred, v_target)

        # For the linear FM path, x_0 = x_1 - v_target, so x_1_hat = x_0 + v_pred.
        endpoint_pred = target + (v_pred - v_target)
        endpoint_pred = torch.clamp(endpoint_pred, min=0.0)

        loss_qmse = self.qmse_loss_fn(endpoint_pred, target, bins)
        loss_psd = self.psd_loss_fn(endpoint_pred, target)
        loss = loss_fm + self.alpha * loss_qmse + self.beta * loss_psd

        return loss, [loss_fm, loss_qmse, loss_psd]
