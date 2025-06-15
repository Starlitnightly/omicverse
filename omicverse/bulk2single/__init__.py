r"""
Bulk-to-single-cell deconvolution and mapping utilities.

This module provides comprehensive tools for bridging bulk and single-cell omics data:
- Converting bulk RNA-seq data to single-cell resolution using deep learning
- Deconvolving cell-type proportions from bulk samples
- Mapping single-cell data to spatial coordinates
- Trajectory blending and temporal analysis
- Variational autoencoder methods for multi-modal data integration

Key capabilities:
- Cross-platform compatibility (bulk RNA-seq, single-cell RNA-seq, spatial)
- Deep learning-based deconvolution algorithms
- Cell type proportion estimation with uncertainty quantification
- Temporal trajectory inference from bulk time-course data
- Integration with popular single-cell analysis workflows

Classes:
    Bulk2Single: Main class for bulk-to-single-cell deconvolution
        - Supports multiple deconvolution algorithms
        - Handles batch effects and technical variations
        - Provides confidence estimates for predictions
        
    Single2Spatial: Single-cell to spatial coordinate mapping
        - Maps single-cell data onto spatial transcriptomics coordinates
        - Preserves spatial relationships and tissue architecture
        - Enables spatially-resolved analysis of single-cell data
        
    BulkTrajBlend: Trajectory blending for temporal analysis
        - Infers developmental trajectories from bulk time-course data
        - Combines bulk and single-cell trajectory information
        - Models continuous developmental processes

Visualization functions:
    bulk2single_plot_cellprop: Plot cell proportion results
    bulk2single_plot_correlation: Plot correlation analysis

Examples:
    >>> import omicverse as ov
    >>> # Bulk to single-cell deconvolution
    >>> bulk2single = ov.bulk2single.Bulk2Single(
    ...     bulk_data=bulk_adata,
    ...     ref_data=sc_adata
    ... )
    >>> bulk2single.train()
    >>> cell_props = bulk2single.predict()
    >>> 
    >>> # Single-cell to spatial mapping
    >>> single2spatial = ov.bulk2single.Single2Spatial(
    ...     sc_data=sc_adata,
    ...     spatial_data=spatial_adata
    ... )
    >>> single2spatial.map_cells()
    >>> 
    >>> # Visualization
    >>> ov.bulk2single.bulk2single_plot_cellprop(cell_props)
"""

from ._bulk2single import Bulk2Single
from ._single2spatial import Single2Spatial
from ._bulktrajblend import BulkTrajBlend
from ._utils import bulk2single_plot_cellprop,bulk2single_plot_correlation

__all__ = [
    'Bulk2Single',
    'Single2Spatial',
    'BulkTrajBlend',
    'bulk2single_plot_cellprop',
    'bulk2single_plot_correlation',
]
