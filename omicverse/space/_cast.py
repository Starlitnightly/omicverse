from scipy.sparse import csr_matrix,issparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

def CAST(adata, sample_key=None, basis='spatial', layer='norm_1e4',
         output_path='output/CAST_Mark', gpu_t=0, device='cuda:0', **kwargs):
    """
    CAST (Cell Annotation for Spatial Transcriptomics) embedding for multiple spatial samples.

    This function implements the CAST algorithm to learn unified embeddings across
    multiple spatial transcriptomics samples, enabling joint analysis and integration
    of spatial data from different sources.

    Arguments:
        adata: AnnData
            Annotated data matrix containing multiple spatial samples.
            Must contain spatial coordinates in adata.obsm[basis].
        sample_key: str, optional (default=None)
            Column name in adata.obs containing sample/batch information.
            Used to identify different samples for integration.
        basis: str, optional (default='spatial')
            Key in adata.obsm containing spatial coordinates.
        layer: str, optional (default='norm_1e4')
            Layer in adata.layers containing normalized expression data.
            Should contain values suitable for CAST analysis.
        output_path: str, optional (default='output/CAST_Mark')
            Directory path for saving CAST intermediate outputs and results.
        gpu_t: int, optional (default=0)
            GPU device index to use for computation.
        device: str, optional (default='cuda:0')
            PyTorch device specification ('cuda:0', 'cpu', etc.).
        **kwargs:
            Additional arguments passed to CAST_MARK function:
            - learning_rate: float
            - n_epochs: int
            - batch_size: int
            - etc.

    Returns:
        AnnData
            Input AnnData object updated with CAST results:
            - adata.obsm['X_cast']: CAST embeddings matrix (n_cells × 512)

    Notes:
        - Requires normalized expression data in specified layer
        - GPU acceleration is enabled by default
        - Creates output directory if it doesn't exist
        - Progress is shown with tqdm progress bars

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load multiple samples
        >>> adata = sc.read_h5ad('spatial_samples.h5ad')
        >>> # Run CAST integration
        >>> adata = ov.space.CAST(
        ...     adata,
        ...     sample_key='sample_id',
        ...     basis='spatial',
        ...     layer='normalized_counts'
        ... )
        >>> # Access CAST embeddings
        >>> cast_embeddings = adata.obsm['X_cast']
    """
    if issparse(adata.obsm[basis]):
        adata.obsm[basis] = adata.obsm[basis].toarray()
    adata.obs['x'] = adata.obsm[basis][:,0]
    adata.obs['y'] = adata.obsm[basis][:,1]

    # Get the coordinates and expression data for each sample
    samples = np.unique(adata.obs[sample_key]) # used samples in adata
    coords_raw = {sample_t: np.array(adata.obs[['x','y']])[adata.obs[sample_key] == sample_t] for sample_t in samples}
    exp_dict = {sample_t: adata[adata.obs[sample_key] == sample_t].layers[layer] for sample_t in samples}

    os.makedirs(output_path, exist_ok=True)
    
    from ..externel.CAST import CAST_MARK
    embed_dict = CAST_MARK(coords_raw, exp_dict, output_path, gpu_t=gpu_t, device=device, **kwargs)

    adata.obsm['X_cast'] = np.zeros((adata.shape[0], 512))
    
    adata.obsm['X_cast'] = pd.DataFrame(adata.obsm['X_cast'], index=adata.obs.index)
    for key in tqdm(embed_dict.keys()):
        adata.obsm['X_cast'].loc[adata.obs[sample_key]==key] += embed_dict[key].cpu().numpy()
    adata.obsm['X_cast'] = adata.obsm['X_cast'].values
    print('CAST embedding is saved in adata.obsm[\'X_cast\']')
    return adata