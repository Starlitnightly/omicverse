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

# External modules
from . import external

# Optional modules
try:
    from . import llm
except ImportError:
    llm = None

# Optional datacollect module
try:
    from .external import datacollect
except ImportError:
    datacollect = None

# Essential utilities - keep these as direct imports for compatibility
from .utils._data import read
from .utils._plot import palette, ov_plot_set, plot_set

# Function registry system for discovery and search
from .utils.registry import (
    find_function,
    list_functions,
    get_function_help,
    recommend_function,
    export_registry,
    import_registry
)

# Smart Agent system using Pantheon
from .utils.smart_agent import Agent, list_supported_models

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