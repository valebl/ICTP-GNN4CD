import numpy as np
from .registry import register_predictor_transform
from utils.helpers.tools import get_backend


@register_predictor_transform("zscore_high_independent")
def transform_zscore_high_independent(x_high, stats=None):
    """
    Independent z-score per feature.
    """
    xp = get_backend(x_high)

    if stats is None:
        # Compute per-channel means/stds using backend ops
        means = xp.array([xp.nanmean(x_high[:, i]) for i in range(x_high.shape[1])])
        stds  = xp.array([xp.nanstd(x_high[:, i])  for i in range(x_high.shape[1])])

        stats = {"means_high": means, "stds_high": stds}
    else:
        means = stats["means_high"]
        stds  = stats["stds_high"]

    # Backend-safe copy
    if hasattr(x_high, "clone"):   # torch tensor
        x = x_high.clone()
    else:                          # numpy array
        x = x_high.copy()

    # Apply independent z-score per feature
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - means[i]) / stds[i]

    return x, stats
