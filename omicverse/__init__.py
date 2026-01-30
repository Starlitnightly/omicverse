r"""
OmicVerse: A comprehensive omic framework for multi-omic analysis.

OmicVerse is a Python package that provides a unified framework for analyzing 
multi-omic data including single-cell RNA-seq, bulk RNA-seq, spatial transcriptomics,
ATAC-seq, and multi-modal integration. It offers streamlined workflows for
preprocessing, analysis, visualization, and interpretation of complex biological data.

Main modules:
    bulk: Bulk RNA-seq analysis including differential expression, pathway analysis
    single: Single-cell RNA-seq analysis including clustering, trajectory inference
    bulk2single: Deconvolution and mapping between bulk and single-cell data
    space: Spatial transcriptomics analysis and integration
    pp: Preprocessing utilities for quality control and normalization
    pl: Comprehensive plotting and visualization functions
    utils: Utility functions for data handling and analysis
    popv: Population-level variation analysis tools

Key features:
    - Unified API for multiple omics data types
    - GPU acceleration support for large-scale analysis
    - Extensive visualization capabilities
    - Integration with popular bioinformatics tools
    - Comprehensive documentation and tutorials

Examples:
    >>> import omicverse as ov
    >>> adata = ov.read('data.h5ad')
    >>> 
    >>> # Traditional approach
    >>> ov.pp.preprocess(adata)
    >>> ov.single.leiden(adata)
    >>> ov.pl.umap(adata, color='leiden')
    >>> 
    >>> # Smart Agent approach  
    >>> agent = ov.Agent(model="gpt-4o-mini", api_key="your-key")
    >>> adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    >>> adata = agent.run("preprocess with 2000 HVGs", adata)
    >>> adata = agent.run("leiden clustering resolution=1.0", adata)
"""

# Fix PyArrow compatibility issue
# PyExtensionType was renamed to ExtensionType in newer versions
import os
try:
    import pyarrow
    if hasattr(pyarrow, 'ExtensionType') and not hasattr(pyarrow, 'PyExtensionType'):
        pyarrow.PyExtensionType = pyarrow.ExtensionType
except ImportError:
    pass

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

# Core submodules - direct imports
from . import alignment
from . import bulk
from . import single
from . import utils
from . import bulk2single
from . import pp
from . import space
from . import pl
from . import datasets
from . import utils

# External modules
from . import external

# Optional modules
if os.environ.get("OMICVERSE_DISABLE_LLM") == "1":
    llm = None
else:
    try:
        from . import llm
    except Exception:
        llm = None


# Optional datacollect module
try:
    from .external import datacollect
except ImportError:
    datacollect = None

# Essential utilities - keep these as direct imports for compatibility
from .utils._data import read
from .utils._plot import palette, ov_plot_set, plot_set, style

# Function registry system for discovery and search
from .utils.registry import (
    find_function,
    list_functions,
    get_function_help,
    recommend_function,
    export_registry,
    import_registry
)

# Smart Agent system (internal backend)
from .utils.smart_agent import Agent, list_supported_models
from .utils.session_notebook_executor import setup_kernel_for_env

name = "omicverse"
try:
    __version__ = version(name)
except Exception:
    __version__ = "unknown"

from ._settings import settings, generate_reference_table

# Common libraries - direct imports for convenience
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData,concat

# Expose agent helpers (e.g., ov.agent.seeker)
try:
    from . import agent  # noqa: F401
except Exception:  # pragma: no cover - optional
    agent = None


__all__ = [
    "alignment",
    "bulk",
    "single",
    "utils",
    "bulk2single",
    "pp",
    "space",
    "pl",
    "datasets",
    "external",
    "llm",
    "datacollect",

    "read",
    "palette",
    "ov_plot_set",
    "plot_set",
    "style",
    "find_function",
    "list_functions",
    "get_function_help",
    "recommend_function",
    "export_registry",
    "import_registry",

    "Agent",
    "list_supported_models",
    "setup_kernel_for_env",
    "settings",
    "generate_reference_table",
    "plt",
    "np",
    "pd",
    "AnnData",
    "concat",
    "agent",

    "__version__",
]