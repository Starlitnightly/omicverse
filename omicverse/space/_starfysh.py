from ..externel.starfysh import (AA, utils, plot_utils, post_analysis)
from ..externel.starfysh import _starfysh as sf_model

class STARFYSH:
    """
    STARFYSH spatial transcriptomics analysis class.
    
    A comprehensive tool for analyzing spatial transcriptomics data, providing
    advanced capabilities for spatial pattern detection and cell type deconvolution.

    Arguments:
        adata: AnnData
            Annotated data matrix containing spatial transcriptomics data.
            Must contain spatial coordinates in adata.obsm['spatial'].

    Attributes:
        adata: AnnData
            The input AnnData object containing spatial data.

    Notes:
        - STARFYSH requires properly formatted spatial transcriptomics data
        - The input AnnData object should contain normalized gene expression data
        - Spatial coordinates should be stored in adata.obsm['spatial']

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load spatial transcriptomics data
        >>> adata = sc.read_h5ad('spatial_data.h5ad')
        >>> # Initialize STARFYSH object
        >>> starfysh = ov.space.STARFYSH(adata)
    """

    def __init__(self, adata):
        """
        Initialize STARFYSH analysis object.
        
        Arguments:
            adata: AnnData
                Annotated data matrix containing spatial transcriptomics data.
                Must contain spatial coordinates in adata.obsm['spatial'].
        """
        pass