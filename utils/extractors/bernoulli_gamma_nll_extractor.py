from .registry import register_extractor
import torch


@register_extractor("Bernoulli_Gamma_NLL_Loss")
def extract_bernoulli_gamma_mean(y_out, threshold_nll, **kwargs):
    """
    Extracts (mean, p, shape, scale) from raw model output.

    Must match Bernoulli_Gamma_NLL_Loss's parametrization exactly:
      - shape/scale via exp() (not softplus)
      - `threshold_nll`: the loss shifts the target down by this amount before
        fitting the Gamma branch, so shape*scale is the mean of the EXCESS
        over threshold_nll given rain, not the mean precipitation amount
        itself. The true conditional mean given rain is therefore
        threshold_nll + shape*scale. Keep this value in sync with whatever
        `threshold_nll` the loss was trained with.
    """

    p_raw = y_out[:, 0]
    shape_raw = y_out[:, 1]
    scale_raw = y_out[:, 2]

    p = torch.sigmoid(p_raw)
    shape = torch.exp(shape_raw)
    scale = torch.exp(scale_raw)

    conditional_mean = shape * scale
    if threshold_nll is not None:
        conditional_mean = conditional_mean + threshold_nll

    mean = p * conditional_mean
    return mean, p, shape, scale

# @register_extractor("Bernoulli_Gamma_NLL_Loss")
# def extract_bernoulli_gamma_sample(y_out, threshold: float | None = 0.1):
#     """
#     Draws a sample from the Bernoulli-Gamma mixture, matching
#     Bernoulli_Gamma_NLL_Loss's parametrization exactly (see
#     extract_bernoulli_gamma_mean for details). Use this for PDF /
#     distributional comparisons; use the mean extractor for point-accuracy
#     metrics (MAE, bias maps).
#     """
#     p_raw = y_out[:, 0]
#     shape_raw = y_out[:, 1]
#     scale_raw = y_out[:, 2]

#     p = torch.sigmoid(p_raw)
#     shape = torch.exp(shape_raw)
#     scale = torch.exp(scale_raw)

#     wet = torch.bernoulli(p)
#     gamma_sample = torch.distributions.Gamma(shape, 1.0 / scale).sample()  # excess over threshold
#     amount_if_wet = gamma_sample + (threshold if threshold is not None else 0.0)

#     return wet * amount_if_wet, p, shape, scale