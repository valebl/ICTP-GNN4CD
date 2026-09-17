import numpy as np
from .registry import get_predictand_transform

def transform_predictand(values, mode, train_idxs, stats_save_path):
    """
    Apply transformation to the predictand (target or model output)
    Saves stats when needed.
    """
    print(mode)
    if mode == "none":
        stats = {"mode": mode}
        if stats_save_path is not None:
            np.savez(stats_save_path, **stats)
        return values

    # Computes stats and transform accordingly
    values_std, stats_values = get_predictand_transform(mode)(values, train_idxs)

    stats = {**stats_values, "mode": mode}

    if stats_save_path is not None:
        np.savez(stats_save_path, **stats)

    return values_std
