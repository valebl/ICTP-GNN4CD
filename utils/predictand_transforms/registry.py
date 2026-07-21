# utils/predictand_transforms/registry.py

PREDICTAND_TRANSFORM_REGISTRY = {}
PREDICTAND_INVERSE_TRANSFORM_REGISTRY = {}

def register_predictand_transform(mode):
    """Decorator for forward transforms."""
    def decorator(func):
        if mode in PREDICTAND_TRANSFORM_REGISTRY and PREDICTAND_TRANSFORM_REGISTRY[mode] is not func:
            raise ValueError(
                f"Predictand transform '{mode}' is already registered to "
                f"{PREDICTAND_TRANSFORM_REGISTRY[mode]!r}; refusing to overwrite with {func!r}. "
                f"Check for duplicate registrations."
            )
        PREDICTAND_TRANSFORM_REGISTRY[mode] = func
        return func
    return decorator

def register_predictand_inverse_transform(mode):
    """Decorator for inverse transforms."""
    def decorator(func):
        if mode in PREDICTAND_INVERSE_TRANSFORM_REGISTRY and PREDICTAND_INVERSE_TRANSFORM_REGISTRY[mode] is not func:
            raise ValueError(
                f"Predictand inverse transform '{mode}' is already registered to "
                f"{PREDICTAND_INVERSE_TRANSFORM_REGISTRY[mode]!r}; refusing to overwrite with {func!r}. "
                f"Check for duplicate registrations."
            )
        PREDICTAND_INVERSE_TRANSFORM_REGISTRY[mode] = func
        return func
    return decorator

def get_predictand_transform(mode):
    if mode not in PREDICTAND_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown predictand transform: {mode}")
    return PREDICTAND_TRANSFORM_REGISTRY[mode]

def get_predictand_inverse_transform(mode):
    if mode not in PREDICTAND_INVERSE_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown predictand inverse transform: {mode}")
    return PREDICTAND_INVERSE_TRANSFORM_REGISTRY[mode]