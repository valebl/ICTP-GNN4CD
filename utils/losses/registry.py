LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(cls):
        """
        cls: the object being decorated
        """
        if name in LOSS_REGISTRY and LOSS_REGISTRY[name] is not cls:
            raise ValueError(
                f"Loss '{name}' is already registered to {LOSS_REGISTRY[name]!r}; "
                f"refusing to overwrite with {cls!r}. Check for duplicate "
                f"registrations (e.g. a backup/old copy of this loss file "
                f"being imported alongside the current one)."
            )
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator

def get_loss(name):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_REGISTRY[name]