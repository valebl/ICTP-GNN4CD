from .registry import register_extractor

@register_extractor("MultiVariable_CRPS_Loss")
def extract_multivariable_crps_ensemble(y_out, **kwargs):
    # y_out: (M, N, n_target_variables, output_dim) ensemble predictions
    # (e.g. from model.generate_ensemble() or forward() in training mode),
    # same convention as extract_crps_ensemble but with the extra
    # n_target_variables axis produced by the multivariable model    
    if y_out.shape[-1] == 1:
        return y_out.squeeze(-1)  # (M, N, n_target_variables)
    return y_out