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

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

# Core submodules
from . import bulk, single, utils, bulk2single, pp, space, pl

# Optional PopV module requires ``scvi-tools``. Skip if dependency missing.

#usually
from .utils._data import read
from .utils._plot import palette,ov_plot_set,plot_set


name = "omicverse"
try:
    __version__ = version(name)
except Exception:
    __version__ = "unknown"


from ._settings import settings,generate_reference_table


# 导入 matplotlib.pyplot
import matplotlib.pyplot as plt

# 将 plt 作为 omicverse 的一个属性
plt = plt  # 注意：确保没有其他变量名冲突

import numpy as np

np = np  # 注意：确保没有其他变量名冲突

# 导入 pandas
import pandas as pd

# 将 pd 作为 omicverse 的一个属性
pd = pd  # 注意：确保没有其他变量名冲突