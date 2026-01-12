r"""
Spatial transcriptomics analysis utilities.

This module provides comprehensive tools for analyzing spatial transcriptomics data
including spatial clustering, integration, trajectory analysis, and visualization.
Supported platforms include 10x Visium, Slide-seq, MERFISH, seqFISH, and others.

Spatial clustering and segmentation:
    pySTAGATE: Graph attention networks for spatial domains
    clusters: Spatial domain identification
    merge_cluster: Merge spatial clusters
    CAST: Cellular spatial organization analysis
    
Spatial integration and alignment:
    pySTAligner: Multi-sample spatial alignment
    Cal_Spatial_Net: Spatial neighborhood networks
    pySpaceFlow: Spatial flow analysis
    
Spatially variable genes:
    svg: Identify spatially variable genes
    
Spatial trajectory analysis:
    STT: Spatial transition tensor analysis
    CellMap: Cell mapping in spatial context
    CellLoc: Cell localization analysis
    
Spatial deconvolution:
    Tangram: Map single-cell data to spatial coordinates
    GASTON: Spatial deconvolution and analysis
    
Utility functions:
    Various spatial analysis tools and helper functions

Examples:
    >>> import omicverse as ov
    >>> import scanpy as sc
    >>> 
    >>> # Load spatial data
    >>> adata = sc.read_visium('path/to/visium/data')
    >>> 
    >>> # Identify spatial domains
    >>> ov.space.pySTAGATE(adata, n_domains=7, radius=50)
    >>> 
    >>> # Find spatially variable genes
    >>> adata = ov.space.svg(adata, mode='prost', n_svgs=2000)
    >>> 
    >>> # Spatial trajectory analysis
    >>> stt = ov.space.STT(adata, spatial_loc='spatial')
    >>> stt.stage_estimate()
    >>> stt.train(n_states=10)
    >>> 
    >>> # Spatial integration
    >>> aligner = ov.space.pySTAligner(adata_list)
    >>> aligner.spatial_alignment()
"""

from ._cluster import pySTAGATE,clusters,merge_cluster
from ._tangram import Tangram
from ._integrate import pySTAligner,Cal_Spatial_Net
from ._spaceflow import pySpaceFlow
from ._spatrio import CellMap,CellLoc
from ._stt import STT
from ._svg import svg
from ._cast import CAST
from ._tools import *
from ._gaston import GASTON
from ._commot import create_communication_anndata,update_classification_from_database
from ._deconvolution import Deconvolution,calculate_gene_signature


__all__ = [
    # Spatial clustering and domains
    'pySTAGATE',
    'clusters',
    'merge_cluster',
    'CAST',
    
    # Spatial integration and mapping
    'Tangram',
    'pySTAligner',
    'Cal_Spatial_Net',
    'pySpaceFlow',
    
    # Spatially variable genes
    'svg',
    
    # Spatial trajectory and dynamics
    'STT',
    'CellMap',
    'CellLoc',
    
    # Spatial deconvolution
    'GASTON',
    
    # Utility functions (imported from _tools)
    'create_communication_anndata',
    'update_classification_from_database',

    # Spatial deconvolution
    'Deconvolution',
    'calculate_gene_signature',
]
