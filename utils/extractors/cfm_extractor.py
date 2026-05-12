from .registry import register_extractor

@register_extractor("CFM_Loss")
def extract_cfm_samples_mean(out):
    if type(out) == list:
        y = out[0]
    else:
        y = out 
    return y
