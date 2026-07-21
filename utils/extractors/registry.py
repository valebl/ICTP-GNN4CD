EXTRACTOR_REGISTRY = {}

def register_extractor(loss_name):
    def decorator(func):
        if loss_name in EXTRACTOR_REGISTRY and EXTRACTOR_REGISTRY[loss_name] is not func:
            raise ValueError(
                f"Extractor for '{loss_name}' is already registered to "
                f"{EXTRACTOR_REGISTRY[loss_name]!r}; refusing to overwrite with {func!r}. "
                f"Check for duplicate registrations."
            )
        EXTRACTOR_REGISTRY[loss_name] = func
        return func
    return decorator

def get_extractor(name):
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor: {name}")
    return EXTRACTOR_REGISTRY[name]