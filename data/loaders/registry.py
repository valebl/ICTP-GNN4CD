DATASET_LOADER_REGISTRY = {}

def register_dataset_loader(name):
    def decorator(func):
        if name in DATASET_LOADER_REGISTRY and DATASET_LOADER_REGISTRY[name] is not func:
            raise ValueError(
                f"Dataset loader '{name}' is already registered to "
                f"{DATASET_LOADER_REGISTRY[name]!r}; refusing to overwrite with {func!r}. "
                f"Check for duplicate registrations."
            )
        DATASET_LOADER_REGISTRY[name] = func
        return func
    return decorator

def get_dataset_loader(name):
    if name not in DATASET_LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset loader: {name}")
    return DATASET_LOADER_REGISTRY[name]