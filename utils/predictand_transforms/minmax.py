import numpy as np
import torch
from .registry import register_predictand_transform, register_predictand_inverse_transform
from utils.helpers.tools import get_backend


@register_predictand_transform("minmax")
def transform_minmax(values, train_idxs=None):
    xp = get_backend(values)

    if train_idxs is not None:
        values_train = values[:, train_idxs]
    else:
        values_train = values

    min_val = xp.floor(xp.nanmin(values_train))
    max_val = xp.ceil(xp.nanmax(values_train))

    stats = {"min": min_val, "max": max_val}

    return (values - min_val) / (max_val - min_val), stats


@register_predictand_inverse_transform("minmax")
def inverse_transform_minmax(values_norm, stats):
    return values_norm * (stats["max"] - stats["min"]) + stats["min"]

