import numpy as np
from .registry import get_predictand_inverse_transform

def inverse_transform_predictand(values_norm, stats):
    """
    Apply inverse transformation to model outputs or normalized targets.
    Loads stats saved during training.
    """
    
    mode = stats["mode"].item()
    if isinstance(mode, np.ndarray):
        mode = mode.item()
    
    if mode == "none":
        return values_norm
    
    values = get_predictand_inverse_transform(mode)(values_norm, stats)

    return values