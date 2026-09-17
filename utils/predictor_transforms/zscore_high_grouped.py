# utils/predictor_transforms/highres_zscore_grouped.py

import numpy as np
import torch
from .registry import register_predictor_transform
from utils.helpers.tools import get_backend


@register_predictor_transform("zscore_high_grouped")
def transform_highres_zscore_grouped(x_high, stats=None):
    """
    Grouped z-score:
    - channel 0 separate # orography
    - channels 1-N grouped together # land use
    """
    xp = get_backend(x_high)

    if stats is None:
        if x_high.shape[1] > 1:
            mean_0 = xp.nanmean(x_high[:, 0])
            mean_grouped = xp.nanmean(x_high[:, 1:])
            std_0 = xp.nanstd(x_high[:, 0])
            std_grouped = xp.nanstd(x_high[:, 1:])

            # store as backend-native arrays/tensors
            means = xp.array([mean_0, mean_grouped])
            stds  = xp.array([std_0, std_grouped])
        else:
            means = xp.nanmean(x_high)
            stds  = xp.nanstd(x_high)

        stats = {"means_high": means, "stds_high": stds}
    else:
        means = stats["means_high"]
        stds  = stats["stds_high"]

    # backend-safe copy
    x = x_high.clone() if isinstance(x_high, torch.Tensor) else x_high.copy()

    if x.shape[1] > 1:
        x[:, 0]  = (x[:, 0]  - means[0]) / stds[0]
        x[:, 1:] = (x[:, 1:] - means[1]) / stds[1]
    else:
        x = (x - means) / stds

    return x, stats

