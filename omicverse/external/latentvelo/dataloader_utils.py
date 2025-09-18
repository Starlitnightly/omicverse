"""Utilities for handling AnnData views in dataloaders"""

def handle_adata_view(adata):
    """
    Handle AnnData views by converting them to regular AnnData objects.
    This avoids COO matrix issues when trying to modify views.
    """
    if not adata.is_view:
        return adata

    import anndata as ad
    from scipy.sparse import issparse, coo_matrix

    # For views, we need to get the parent data and subset it manually
    # This avoids the COO matrix subscripting issue
    parent = adata._adata_ref
    oidx = adata._oidx
    vidx = adata._vidx

    # Extract X matrix, converting COO to CSR if necessary
    X = parent.X
    if isinstance(X, coo_matrix):
        X = X.tocsr()

    # Subset X using the indices
    if oidx is not None and vidx is not None:
        X = X[oidx, :][:, vidx]
    elif oidx is not None:
        X = X[oidx, :]
    elif vidx is not None:
        X = X[:, vidx]

    # Get obs and var from the view (these are already subsetted)
    obs = adata.obs.copy()
    var = adata.var.copy()

    # Create new AnnData
    adata_new = ad.AnnData(X, obs=obs, var=var)

    # Copy layers, converting COO to CSR if needed and subsetting manually
    for key in parent.layers.keys():
        layer = parent.layers[key]
        if isinstance(layer, coo_matrix):
            layer = layer.tocsr()

        # Subset layer using the same indices as X
        if oidx is not None and vidx is not None:
            layer = layer[oidx, :][:, vidx]
        elif oidx is not None:
            layer = layer[oidx, :]
        elif vidx is not None:
            layer = layer[:, vidx]

        adata_new.layers[key] = layer

    # Copy other attributes
    for key in adata.obsm.keys():
        adata_new.obsm[key] = adata.obsm[key].copy() if hasattr(adata.obsm[key], 'copy') else adata.obsm[key]

    # Handle obsp carefully - subset from parent to avoid COO matrix access issues
    for key in parent.obsp.keys():
        val = parent.obsp[key]
        if isinstance(val, coo_matrix):
            val = val.tocsr()

        # Subset using observation indices only (both dimensions use oidx for cell-cell matrices)
        if oidx is not None:
            val = val[oidx, :][:, oidx]

        adata_new.obsp[key] = val

    for key in adata.uns.keys():
        adata_new.uns[key] = adata.uns[key]

    return adata_new


def ensure_csr_csc(adata):
    """
    Ensure sparse matrices in `adata` are in CSR/CSC formats to allow safe subsetting.
    - Converts `.X` and all `layers` to CSR if they are COO.
    - Converts `obsp` entries to CSR if they are COO.
    The operation is in-place and returns the same `adata` for convenience.
    """
    from scipy.sparse import coo_matrix, issparse

    # X matrix
    X = adata.X
    if isinstance(X, coo_matrix):
        adata.X = X.tocsr()

    # Layers
    for k in list(adata.layers.keys()):
        v = adata.layers[k]
        if isinstance(v, coo_matrix):
            adata.layers[k] = v.tocsr()

    # obsp matrices
    for k in list(adata.obsp.keys()):
        v = adata.obsp[k]
        if isinstance(v, coo_matrix):
            adata.obsp[k] = v.tocsr()

    return adata
