r"""
Dataset downloading and management utilities.

This module provides functions to automatically download and process datasets
commonly used in single-cell and spatial omics analysis, particularly those
from scanpy tutorials and other standard benchmarking datasets.

Main functions:
    load_scanpy_pbmc3k: Load PBMC 3k dataset from scanpy
    load_scanpy_pbmc68k: Load PBMC 68k dataset from scanpy  
    load_clustering_tutorial_data: Load data from scanpy clustering tutorial
    download_and_cache: Generic download and caching utility
    
Examples:
    >>> import omicverse as ov
    >>> 
    >>> # Load PBMC 3k data for testing
    >>> adata = ov.datasets.load_scanpy_pbmc3k(processed=True)
    >>> 
    >>> # Load data from clustering tutorial
    >>> adata = ov.datasets.load_clustering_tutorial_data()
    >>> 
    >>> # Use in statistical functions
    >>> ov.utils.shannon_diversity(adata, 'sample', 'cell_type')
"""

from ._datasets import (
    load_scanpy_pbmc3k,
    load_scanpy_pbmc68k, 
    load_clustering_tutorial_data,
    download_and_cache,
    list_available_datasets,
    create_mock_dataset
)