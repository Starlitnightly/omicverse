r"""
Plotting utilities for omics data visualization.

This module provides comprehensive plotting functions for visualizing various types
of omics data including single-cell, bulk, spatial transcriptomics, multi-omics,
and network data. It offers publication-ready plots with customizable aesthetics.

Visualization categories:
    Single-cell plots: UMAP, t-SNE, PCA, violin plots, dot plots
    Bulk data plots: Heatmaps, volcano plots, MA plots, PCA plots
    Spatial plots: Spatial feature plots, domain visualization
    Multi-omics plots: Factor plots, integration visualizations
    Network plots: Protein-protein interactions, pathway networks
    General plots: Heatmaps, scatter plots, bar plots
    
Key modules:
    _single: Single-cell specific plotting functions
    _bulk: Bulk RNA-seq plotting functions  
    _space: Spatial transcriptomics visualizations
    _multi: Multi-omics integration plots
    _heatmap: Heatmap and clustering visualizations
    _general: General-purpose plotting utilities
    _palette: Color palette and aesthetic functions
    _cpdb: Cell-cell communication network plots
    _flowsig: Flow cytometry-style visualizations
    _embedding: Dimensionality reduction visualizations
    _density: Density and distribution plots

Features:
    - Publication-ready figures with customizable aesthetics
    - Integration with matplotlib and seaborn
    - Support for interactive plots
    - Consistent color schemes and themes
    - Export capabilities for various formats
    - Integration with AnnData objects

Examples:
    >>> import omicverse as ov
    >>> # Single-cell visualization
    >>> ov.pl.embedding(adata, basis='umap', color='cell_type')
    >>> ov.pl.violin(adata, keys=['CD3D', 'CD8A'], groupby='cell_type')
    >>> 
    >>> # Bulk data visualization  
    >>> ov.pl.volcano(deg_results, pval_threshold=0.05)
    >>> ov.pl.heatmap(adata, var_names=marker_genes)
    >>> 
    >>> # Spatial visualization
    >>> ov.pl.spatial(adata, color='total_counts')
    >>> ov.pl.spatial_domains(adata, color='domain')
"""
from ._palette import *
from ._single import *
from ._general import *
from ._heatmap import *
from ._multi import *
from ._bulk import *
from ._space import *
from ._cpdb import *
from ._flowsig import *
from ._embedding import *
from ._density import *
from ._cpdbviz import *
from ._dotplot import dotplot, rank_genes_groups_dotplot
from ._spatial import spatial_segment,spatial_segment_overlay
from ._violin import violin

# Note: Specific function names are imported through wildcard imports
# from individual modules. Key functions include:
# - embedding, umap, tsne: dimensionality reduction plots
# - violin, dotplot, stacked_violin: expression distribution plots  
# - spatial, spatial_domains: spatial transcriptomics plots
# - heatmap, clustermap: heatmap visualizations
# - volcano, ma_plot: differential expression plots
# - network, chord_diagram: network visualizations
# - palette functions: color scheme utilities

