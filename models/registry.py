MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        if name in MODEL_REGISTRY and MODEL_REGISTRY[name] is not cls:
            raise ValueError(
                f"Model '{name}' is already registered to {MODEL_REGISTRY[name]!r}; "
                f"refusing to overwrite with {cls!r}. Check for duplicate registrations."
            )
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]