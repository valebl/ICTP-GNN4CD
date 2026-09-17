import pkgutil
import importlib
from .registry import (
    PREDICTOR_TRANSFORM_REGISTRY,
    register_predictor_transform,
    get_predictor_transform
)

# Automatically import all modules in this package
for module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_info.name}")