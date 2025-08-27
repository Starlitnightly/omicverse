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
    >>> ov.pp.preprocess(adata)
    >>> ov.single.leiden(adata)
    >>> ov.pl.umap(adata, color='leiden')
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

# Lazy loading system for faster imports
from ._lazy_loader import create_lazy_module, create_lazy_attribute

# Core submodules - loaded lazily to improve import speed
bulk = create_lazy_module('omicverse.bulk', globals())
single = create_lazy_module('omicverse.single', globals())
utils = create_lazy_module('omicverse.utils', globals())
bulk2single = create_lazy_module('omicverse.bulk2single', globals())
pp = create_lazy_module('omicverse.pp', globals())
space = create_lazy_module('omicverse.space', globals())
pl = create_lazy_module('omicverse.pl', globals())
llm = create_lazy_module('omicverse.llm', globals())
datasets = create_lazy_module('omicverse.datasets', globals())

# External modules - loaded lazily  
external = create_lazy_module('omicverse.external', globals())

# Optional datacollect module
try:
    # Test if datacollect is available by attempting import
    import omicverse.external.datacollect
    datacollect = create_lazy_module('omicverse.external.datacollect', globals())
except ImportError:
    # Gracefully handle missing datacollect module
    datacollect = None

# Essential utilities - keep these as direct imports for compatibility
from .utils._data import read
from .utils._plot import palette, ov_plot_set, plot_set

name = "omicverse"
try:
    __version__ = version(name)
except Exception:
    __version__ = "unknown"

from ._settings import settings, generate_reference_table

# Heavy libraries - loaded lazily to improve import speed
# These will only be imported when first accessed
plt = create_lazy_attribute('matplotlib.pyplot')
np = create_lazy_attribute('numpy')  
pd = create_lazy_attribute('pandas')