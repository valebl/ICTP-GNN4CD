import torch.nn as nn
from utils.loss_functions.psd import PSDLoss
from utils.loss_functions.qmse import QMSELoss


class MSE_QMSE_PSD_Loss(nn.Module):
    def __init__(
        self,
        alpha,
        beta,
        balance=None,
        *psd_args,
        **psd_kwargs
        ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.mse_loss_fn = nn.MSELoss()
        self.qmse_loss_fn = QMSELoss(balance)
        self.psd_loss_fn = PSDLoss(apply_expm1=True, *psd_args, **psd_kwargs)

    def forward(self, pred, target, bins):
        loss_mse = self.mse_loss_fn(pred, target)
        loss_qmse = self.qmse_loss_fn(pred, target, bins)
        loss_psd = self.psd_loss_fn(pred, target)
        loss = loss_mse + self.alpha * loss_qmse + self.beta * loss_psd
        return loss, loss_mse, loss_qmse, loss_psd



    