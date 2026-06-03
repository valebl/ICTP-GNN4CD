from .registry import register_extractor
import torch.nn.functional as F
import torch

@register_extractor("Bernoulli_Gamma_NLL_Loss")
def extract_bernoulli_gamma_mean(y_out):
    p_raw = y_out[:, 0]
    shape_raw = y_out[:, 1]
    scale_raw = y_out[:, 2]

    # transforms
    p = torch.sigmoid(p_raw)
    shape = F.softplus(shape_raw)
    scale = F.softplus(scale_raw)

    # mean of mixture: p * mean_gamma
    return p * shape * scale

