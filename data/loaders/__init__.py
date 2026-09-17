import pkgutil
import importlib
from .registry import DATASET_LOADER_REGISTRY, register_dataset_loader, get_dataset_loader

# Automatically import all modules in this package
for module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_info.name}")