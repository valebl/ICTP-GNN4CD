from .registry import register_extractor

@register_extractor("CFM_Loss")
def extract_cfm_samples_mean(out):
    dim = out.size(dim=1)
    y = out[:, dim-1]
    return y
