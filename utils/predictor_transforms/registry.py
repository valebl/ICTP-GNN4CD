PREDICTOR_TRANSFORM_REGISTRY = {}

def register_predictor_transform(mode):
    """Decorator for forward transforms."""
    def decorator(func):
        if mode in PREDICTOR_TRANSFORM_REGISTRY and PREDICTOR_TRANSFORM_REGISTRY[mode] is not func:
            raise ValueError(
                f"Predictor transform '{mode}' is already registered to "
                f"{PREDICTOR_TRANSFORM_REGISTRY[mode]!r}; refusing to overwrite with {func!r}. "
                f"Check for duplicate registrations."
            )
        PREDICTOR_TRANSFORM_REGISTRY[mode] = func
        return func
    return decorator

def get_predictor_transform(mode):
    if mode not in PREDICTOR_TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown predictor transform: {mode}")
    return PREDICTOR_TRANSFORM_REGISTRY[mode]