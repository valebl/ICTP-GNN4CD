from .registry import register_extractor

@register_extractor("MSE_QMSE_PSD_Loss")
def extract_mse_qmse_psd(y_out, **kwargs):
    return y_out.squeeze(-1)