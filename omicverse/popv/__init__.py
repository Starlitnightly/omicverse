r"""
PopV: Population-level Variation analysis for single-cell data.

PopV is a framework for analyzing population-level variation in single-cell data,
focusing on automated cell type annotation, batch integration, and algorithm 
benchmarking. It provides a unified interface for comparing and evaluating
different single-cell analysis methods.

Key features:
- Automated cell type annotation using multiple algorithms
- Comprehensive batch integration methods
- Algorithm benchmarking and evaluation metrics
- Hub for pre-trained models and reference datasets
- Population-level variation analysis
- Integration with major single-cell frameworks

Main modules:
    algorithms: Collection of annotation and integration algorithms
    annotation: Cell type annotation utilities and workflows
    hub: Model hub for pre-trained models and reference data
    preprocessing: Data preprocessing and quality control
    visualization: Plotting and visualization functions

Supported algorithms:
- CellTypist: Automated cell type annotation
- SCANVI: Semi-supervised annotation with variational inference
- Harmony: Fast integration of single-cell data
- SCVI: Single-cell variational inference
- OnClass: Ontology-guided cell type annotation
- And many more through the algorithms module

Examples:
    >>> import omicverse as ov
    >>> # Automated annotation
    >>> ov.popv.annotation.annotate_cell_types(
    ...     adata, 
    ...     model='best_human_model'
    ... )
    >>> 
    >>> # Batch integration
    >>> ov.popv.algorithms.harmony_integrate(
    ...     adata,
    ...     batch_key='batch'
    ... )
    >>> 
    >>> # Model evaluation
    >>> metrics = ov.popv.hub.evaluate_model(
    ...     adata,
    ...     model_name='my_model'
    ... )

Notes:
    - Requires Python 3.10+ for pre-trained model compatibility
    - Some models are hosted on HuggingFace and require internet access
    - Integrates seamlessly with scanpy and other single-cell tools
"""

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
