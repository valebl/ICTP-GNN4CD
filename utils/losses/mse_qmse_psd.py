import torch.nn as nn
from .registry import register_loss
from.psd import PSD_Loss
from.qmse import QMSE_Loss


@register_loss("MSE_QMSE_PSD_Loss")
class MSE_QMSE_PSD_Loss(nn.Module):
    output_dim = 1 # class attribute
    use_bins = True
    components = ["MSE", "QMSE", "PSD"]

    @staticmethod
    def add_loss_specific_args(parser):
        parser.add_argument("--alpha", type=float)
        parser.add_argument("--beta", type=float)
        parser.add_argument('--binmin', type=float)
        parser.add_argument('--binmax', type=float)
        parser.add_argument('--binwidth', type=float)
        parser.add_argument('--binscale', type=str)
        parser.add_argument('--y_dim', type=int)
        parser.add_argument('--x_dim', type=int)   
        return parser

    def __init__(
        self,
        alpha,
        beta,
        balance=None,
        x_dim=128,
        y_dim=128,
        ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss_fn = nn.MSELoss()
        self.qmse_loss_fn = QMSE_Loss(balance)
        self.psd_loss_fn = PSD_Loss(apply_expm1=True, x_dim=x_dim, y_dim=y_dim)

    def forward(self, pred, target, bins):
        pred = pred.squeeze()
        loss_mse = self.mse_loss_fn(pred, target)
        loss_qmse = self.qmse_loss_fn(pred, target, bins)
        loss_psd = self.psd_loss_fn(pred, target)
        loss = loss_mse + self.alpha * loss_qmse + self.beta * loss_psd
        return loss, [loss_mse, loss_qmse, loss_psd]
