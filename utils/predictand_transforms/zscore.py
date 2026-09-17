import numpy as np
import torch
from .registry import register_predictand_transform, register_predictand_inverse_transform
from utils.helpers.tools import get_backend


@register_predictand_transform("zscore")
def transform_zscore(values, train_idxs=None):
    xp = get_backend(values)

    if train_idxs is not None:
        values_train = values[:, train_idxs]
    else:
        values_train = values

    mean = xp.nanmean(values_train)
    std  = xp.nanstd(values_train)

    stats = {"mean": mean, "std": std}

    return (values - mean) / std, stats


@register_predictand_inverse_transform("zscore")
def inverse_transform_zscore(values_norm, stats):
    return values_norm * stats["std"] + stats["mean"]

