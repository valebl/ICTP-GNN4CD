from .registry import register_extractor
import torch


@register_extractor("Bernoulli_Gamma_NLL_Loss")
def extract_bernoulli_gamma_mean(y_out, threshold: float | None = 0.1):
    """
    Extracts (mean, p, shape, scale) from raw model output.

    Must match Bernoulli_Gamma_NLL_Loss's parametrization exactly:
      - shape/scale via exp() (not softplus)
      - `threshold`: the loss shifts the target down by this amount before
        fitting the Gamma branch, so shape*scale is the mean of the EXCESS
        over threshold given rain, not the mean precipitation amount
        itself. The true conditional mean given rain is therefore
        threshold + shape*scale. Keep this value in sync with whatever
        `threshold` the loss was trained with.
    """
    p_raw = y_out[:, 0]
    shape_raw = y_out[:, 1]
    scale_raw = y_out[:, 2]

    p = torch.sigmoid(p_raw)
    shape = torch.exp(shape_raw)
    scale = torch.exp(scale_raw)

    conditional_mean = shape * scale
    if threshold is not None:
        conditional_mean = conditional_mean + threshold

    mean = p * conditional_mean
    return mean, p, shape, scale