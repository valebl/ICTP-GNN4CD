import numpy as np
from .registry import get_predictor_transform

def transform_predictors(
    x_low,
    x_high,
    train_idxs=None,
    mode_low=None,
    mode_high=None,
    stats=None,
    stats_save_path=None,
    **kwargs
):

    if stats is not None:
        mode_low = stats["mode_low"]
        if isinstance(mode_low, np.ndarray):
            mode_low = mode_low.item()
        mode_high = stats["mode_high"]
        if isinstance(mode_high, np.ndarray):
            mode_high = mode_high.item()

    # --- LOW-RES ---
    x_low_std, stats_low = get_predictor_transform(mode_low)(
        x_low, x_low.shape[2], train_idxs, stats
    )

    # --- HIGH-RES ---
    x_high_std, stats_high = get_predictor_transform(mode_high)(
        x_high, stats
    )

    # --- Save unified stats ---
    if stats is None and stats_save_path is not None:

        stats = {**stats_low, **stats_high,
                "mode_low": mode_low,
                "mode_high": mode_high}
        
        np.savez(stats_save_path, **stats)

    return x_low_std, x_high_std
