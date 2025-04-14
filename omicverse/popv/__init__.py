"""PopV."""

import sys
import warnings

if sys.version_info[:2] != (3, 10):
    warnings.warn(
        "Pretrained models on huggingface are trained with Python 3.11. "
        f"Detected Python {sys.version.split()[0]} will not load these models.",
        UserWarning,
        stacklevel=3,
    )

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging

import scanpy as sc

from ._settings import settings  # isort: skip

# Import order to avoid circular imports
from . import algorithms, annotation, hub, preprocessing, visualization
from ._utils import create_ontology_resources

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
package_name = "popv"
__version__ = importlib_metadata.version(package_name)

settings.verbosity = logging.INFO

# Jax sets the root logger, this prevents double output.
popv_logger = logging.getLogger("popv")
popv_logger.propagate = False


__all__ = [
    "algorithms",
    "annotation",
    "create_ontology_resources",
    "hub",
    "preprocessing",
    "settings",
    "visualization",
]
