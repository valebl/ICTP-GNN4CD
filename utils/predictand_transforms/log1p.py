import numpy as np
from .registry import register_predictand_transform, register_predictand_inverse_transform
from utils.helpers.tools import get_backend


@register_predictand_transform("log1p")
def transform_log1p(values, train_idxs=None):
    xp = get_backend(values)
    stats = {}
    return xp.log1p(values), stats

@register_predictand_inverse_transform("log1p")
def inverse_transform_log1p(values_norm, stats):
    xp = get_backend(values_norm)
    return xp.expm1(values_norm)
