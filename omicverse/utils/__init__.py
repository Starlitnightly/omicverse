r"""
Utility functions for data handling and analysis.

This module provides essential utility functions for data I/O, manipulation,
visualization, clustering, and various computational tasks across the omicverse
ecosystem. It serves as the foundation for many higher-level analysis functions.

Data I/O and manipulation:
    read: Universal data reader for various formats
    read_csv, read_10x_mtx, read_h5ad, read_10x_h5: Format-specific readers
    data_downloader: Download datasets and models
    geneset_prepare: Prepare gene sets for analysis
    get_gene_annotation: Gene annotation utilities
    
Visualization utilities:
    palette: Color palette management
    ov_plot_set, plot_set: Global plot settings
    Various plotting helper functions
    
Clustering and analysis:
    cluster: General clustering functions
    LDA_topic: Topic modeling for single-cell data
    filtered, refine_label: Data filtering and label refinement
    
Statistical and computational tools:
    correlation_pseudotime: Pseudotime correlation analysis
    np_mean, np_std: Numpy-based statistical functions
    neighbors: Neighbor graph construction
    mde: Minimum distortion embedding
    
Data structures and conversion:
    anndata_sparse: Sparse matrix utilities for AnnData
    store_layers, retrieve_layers: Layer management
    
Specialized analysis:
    roe: Ratio of expression analysis
    cal_paga, plot_paga: PAGA trajectory analysis
    venny4py: Venn diagram utilities
    syn: Synthetic data generation
    scatterplot: Advanced scatter plot functions
    knn: K-nearest neighbors utilities
    heatmap: Heatmap generation and customization
    lsi: Latent semantic indexing

Examples:
    >>> import omicverse as ov
    >>> # Data reading
    >>> adata = ov.utils.read('data.h5ad')
    >>> 
    >>> # Clustering
    >>> labels = ov.utils.cluster(
    ...     adata.obsm['X_pca'], 
    ...     method='leiden', 
    ...     resolution=0.5
    ... )
    >>> 
    >>> # Visualization setup
    >>> ov.utils.ov_plot_set()
    >>> 
    >>> # Download data
    >>> ov.utils.data_downloader('pbmc3k')
"""

# All functions imported via wildcard imports from submodules
from ._data import *
from ._plot import *
#from ._genomics import *
from ._mde import *
from ._syn import *
from ._scatterplot import *
from ._knn import *
from ._heatmap import *
from ._roe import roe, roe_plot_heatmap
from ._odds_ratio import odds_ratio, plot_odds_ratio_heatmap
from ._shannon_diversity import shannon_diversity, compare_shannon_diversity, plot_shannon_diversity
from ._resolution import optimal_resolution, plot_resolution_optimization, resolution_stability_analysis
from ._paga import cal_paga,plot_paga
from ._cluster import cluster,LDA_topic,filtered,refine_label
from ._venn import venny4py
from ._lsi import *
from ._neighboors import neighbors