from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from datetime import datetime

import numpy as np
from anndata import AnnData

from .._settings import settings, EMOJI, Colors

from ..external.sude_py import sude as sude_py

if TYPE_CHECKING:
    from typing import Literal

def _choose_representation(adata, use_rep=None, n_pcs=None):
    """Simple representation selection without scanpy dependencies."""
    if use_rep is None:
        # Use the main data matrix
        X = adata.X
    elif use_rep in adata.obsm:
        # Use a representation from obsm
        X = adata.obsm[use_rep]
    else:
        raise ValueError(f"Representation '{use_rep}' not found in adata.obsm")
    
    # Apply PCA if requested
    if n_pcs is not None and n_pcs > 0:
        from sklearn.decomposition import PCA
        if X.shape[1] > n_pcs:
            pca = PCA(n_components=n_pcs, random_state=42)
            X = pca.fit_transform(X)
    
    return X


def sude(
    adata: AnnData,
    n_pcs: int | None = None,
    *,
    no_dims: int = 2,
    use_rep: str | None = None,
    k1: int = 20,
    normalize: bool = True,
    large: bool = False,
    initialize: str = "le",
    agg_coef: float = 1.2,
    T_epoch: int = 50,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    r"""SUDE (Scalable Unsupervised Dimensionality reduction via Embedding) dimensionality reduction.

    Perform SUDE dimensionality reduction for visualization of single-cell data.
    SUDE is a scalable unsupervised dimensionality reduction method that can handle
    large-scale datasets efficiently by using landmark sampling and constrained
    locally linear embedding.

    SUDE was proposed for scalable dimensionality reduction of single-cell data.
    It uses a two-stage approach: first computing embeddings for landmark points,
    then interpolating the remaining points using constrained locally linear embedding.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_pcs
        Number of principal components to use for preprocessing.
        If None, uses the full data matrix.
    use_rep
        Key for the representation to use. If None, uses adata.X.
    no_dims
        The number of dimensions of the embedding.
    k1
        Number of nearest neighbors for PPS (Probabilistic Point Sampling) to
        sample landmarks. Must be smaller than the number of observations.
        If set to 0, all points are used as landmarks.
    normalize
        Whether to normalize the data using min-max normalization.
        Should be set to True if features are on different scales.
    large
        Whether to use the large-scale version that splits data into blocks
        to avoid memory overflow. Recommended for datasets with >10k cells.
    initialize
        Method for initializing the embedding before manifold learning.
        Options: 'le' (Laplacian eigenmaps), 'pca' (PCA), 'mds' (MDS).
    agg_coef
        Aggregation coefficient for computing modified distance matrix.
        Controls the influence of shared nearest neighbors.
    T_epoch
        Maximum number of epochs for optimization.
    key_added
        If not specified, the embedding is stored as
        :attr:`~anndata.AnnData.obsm`\ `['X_sude']` and the parameters in
        :attr:`~anndata.AnnData.uns`\ `['sude']`.
        If specified, the embedding is stored as
        :attr:`~anndata.AnnData.obsm`\ ``[key_added]`` and the parameters in
        :attr:`~anndata.AnnData.uns`\ ``[key_added]``.
    copy
        Return a copy instead of writing to `adata`.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields:

    `adata.obsm['X_sude' | key_added]` : :class:`numpy.ndarray` (dtype `float`)
        SUDE coordinates of data.
    `adata.uns['sude' | key_added]` : :class:`dict`
        SUDE parameters.

    Examples
    --------
    >>> import omicverse as ov
    >>> adata = ov.datasets.pbmc3k()
    >>> ov.pp.sude(adata)
    >>> ov.pl.sude(adata, color='leiden')
    """
    print(f"{EMOJI['start']} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running SUDE in '{settings.mode}' mode...")
    print(f"{Colors.CYAN}Computing SUDE dimensionality reduction{Colors.ENDC}")
    
    adata = adata.copy() if copy else adata
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    
    # Validate parameters
    n_obs = X.shape[0]
    if k1 >= n_obs and k1 > 0:
        print(f"{EMOJI['warning']} {Colors.WARNING}k1 ({k1}) is larger than or equal to the number of observations ({n_obs}).{Colors.ENDC}")
        print(f"{Colors.WARNING}    Setting k1 to 0 to use all points as landmarks.{Colors.ENDC}")
        k1 = 0
    
    if initialize not in ['le', 'pca', 'mds']:
        raise ValueError(f"initialize must be one of ['le', 'pca', 'mds'], got '{initialize}'")
    
    if agg_coef <= 0:
        raise ValueError(f"agg_coef must be positive, got {agg_coef}")
    
    if T_epoch <= 0:
        raise ValueError(f"T_epoch must be positive, got {T_epoch}")
    
    # Run SUDE
    print(f"{Colors.BLUE}    Using SUDE dimensionality reduction{Colors.ENDC}")
    print(f"{Colors.GREEN}    Parameters: dims={no_dims}, k1={k1}, normalize={normalize}, large={large}{Colors.ENDC}")
    print(f"{Colors.GREEN}    Initialize={initialize}, agg_coef={agg_coef}, T_epoch={T_epoch}{Colors.ENDC}")
    
    try:
        X_sude = sude_py(
            X,
            no_dims=no_dims,
            k1=k1,
            normalize=normalize,
            large=large,
            initialize=initialize,
            agg_coef=agg_coef,
            T_epoch=T_epoch,
        )
    except Exception as e:
        print(f"{EMOJI['error']} {Colors.FAIL}SUDE failed: {e}{Colors.ENDC}")
        raise

    # Update AnnData instance
    params = dict(
        no_dims=no_dims,
        k1=k1,
        normalize=normalize,
        large=large,
        initialize=initialize,
        agg_coef=agg_coef,
        T_epoch=T_epoch,
        use_rep=use_rep,
    )
    key_uns, key_obsm = ("sude", "X_sude") if key_added is None else [key_added] * 2
    adata.obsm[key_obsm] = X_sude  # annotate samples with SUDE coordinates
    adata.uns[key_uns] = dict(params={k: v for k, v in params.items() if v is not None})

    print(f"{EMOJI['done']} SUDE completed successfully.")
    print(f"{Colors.GREEN}    Added:{Colors.ENDC}")
    print(f"{Colors.CYAN}        {key_obsm!r}, SUDE coordinates (adata.obsm){Colors.ENDC}")
    print(f"{Colors.CYAN}        {key_uns!r}, SUDE parameters (adata.uns){Colors.ENDC}")

    return adata if copy else None
