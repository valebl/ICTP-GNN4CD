from .registry import register_extractor

@register_extractor("Gaussian_NLL_Loss")
def extract_gaussian_nll_mean(y_out, **kwargs):
    mu = y_out[:, 0]
    return mu
