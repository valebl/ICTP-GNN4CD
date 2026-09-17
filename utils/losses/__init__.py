import pkgutil
import importlib
from .registry import LOSS_REGISTRY, register_loss, get_loss

# Automatically import all modules in this package
for module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_info.name}")
