import pkgutil
import importlib
from .registry import (
    PREDICTAND_TRANSFORM_REGISTRY,
    PREDICTAND_INVERSE_TRANSFORM_REGISTRY,
    register_predictand_transform,
    register_predictand_inverse_transform,
    get_predictand_transform,
    get_predictand_inverse_transform
)

# Automatically import all modules in this package
for module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_info.name}")