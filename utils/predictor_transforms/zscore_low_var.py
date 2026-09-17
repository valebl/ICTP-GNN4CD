import numpy as np
from .registry import register_predictor_transform
from utils.helpers.tools import get_backend

@register_predictor_transform("zscore_low_var")
def transform_zscore_low_var(x_low, n_vars, train_idxs=None, stats=None):
    """
    Per-variable z-score aggregated over levels.
    x_low shape: (nodes, time, vars, levels)
    """
    xp = get_backend(x_low)

    # Select training slice
    if train_idxs is not None:
        train_slice = x_low[:, train_idxs, :, :]   # (nodes, time_train, vars, levels)
    else:
        train_slice = x_low

    # Compute or load stats
    if stats is None:
        means = xp.mean(train_slice, axis=(0, 1, 3))   # (n_vars,)
        stds  = xp.std(train_slice,  axis=(0, 1, 3))
        stats = {"means_low": means, "stds_low": stds}
    else:
        means = stats["means_low"]
        stds  = stats["stds_low"]

    # Broadcast shapes: (1, 1, vars, 1)
    means_b = means.reshape(1, 1, n_vars, 1)
    stds_b  = stds.reshape(1, 1, n_vars, 1)

    # Apply transform
    x_low_std = (x_low - means_b) / stds_b

    return x_low_std, stats
