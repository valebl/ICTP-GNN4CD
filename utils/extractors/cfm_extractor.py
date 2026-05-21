from .registry import register_extractor

@register_extractor("CFM_Loss")
@register_extractor("CFM_QMSE_PSD_Loss")
@register_extractor("CFM_AnchorResidual_Loss")
def extract_cfm_samples_mean(out):
    if type(out) == list:
        return out[1]
    return out
