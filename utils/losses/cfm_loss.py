import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_loss


@register_loss("CFM_Loss")
class CFM_Loss(nn.Module):
    output_dim = 1 # class attribute

    def __init__(self):
        super().__init__()

    def forward(self, out, *args, **kwargs):
        """
        Args:
            out: the output of the GNN_CFM_Model or GNN4CD_GrapfCFM_Model
                out[:,0] are the predicted velocities
                out[:,1] are the target velocities
        """
        v_pred = out[:,0]
        v_target = out[:,1]

        return F.mse_loss(v_pred, v_target)
