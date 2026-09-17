from .registry import register_extractor

@register_extractor("CRPS_Loss")
def extract_crps_ensemble(y_out, **kwargs):
    # y_out: (M, N, output_dim) ensemble predictions (e.g. from
    # model.generate_ensemble() or forward() in training mode).
    # Left unreduced on purpose, since the mean must be taken after
    # inverse-transforming each member back to physical units, not here.
    if y_out.shape[-1] == 1:
        return y_out.squeeze(-1)  # (M, N)
    return y_out