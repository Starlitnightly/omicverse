import numpy as np

"""
Adapted from UniTVelo (Gao et al. 2022, bioRxiv)

Selects a set of genes to regularize in gene-space
"""

def get_weight(x, y=None, perc=95):
    from scipy.sparse import issparse

    xy_norm = np.array(x.toarray() if issparse(x) else x)
    if y is not None:
        if issparse(y):
            y = y.A
        xy_norm = xy_norm / np.clip(np.max(xy_norm, axis=0), 1e-3, None)
        xy_norm += y / np.clip(np.max(y, axis=0), 1e-3, None)

    if isinstance(perc, int):
        weights = xy_norm >= np.percentile(xy_norm, perc, axis=0)
    else:
        lb, ub = np.percentile(xy_norm, perc, axis=0)
        weights = (xy_norm <= lb) | (xy_norm >= ub)
    
    return weights

def R2(residual, total):
    r2 = np.ones(residual.shape[1]) - \
        np.sum(residual * residual, axis=0) / \
            np.sum(total * total, axis=0)
    r2[np.isnan(r2)] = 0
    return r2

def compute_velocity_genes(adata_raw, n_top_genes, r2_adjust = True, inplace=True,
                           min_ratio=0.01, min_r2=0.01, max_r2=0.95, perc=[5, 95]):
    
    adata = adata_raw.copy()
    
    fit_offset=False
    vkey='velocity'
    
    Ms = adata_raw.layers['Ms'].copy()
    Mu = adata_raw.layers['Mu'].copy()
    n_obs, n_vars = Ms.shape
    n_var = n_vars

    gamma = np.zeros(n_vars)
    r2 = np.zeros(n_vars)
    velocity_genes = np.ones(n_vars)
    residual_scale = np.zeros([n_obs, n_vars])

    # need get_weight function
    weights = get_weight(Ms, Mu, perc=95)

    Ms_quantile, Mu_quantile = weights * Ms, weights * Mu

    # linear slope using quantiles (weights)
    gamma_quantile = np.sum(Mu_quantile * Ms_quantile, axis=0) / np.sum(Ms_quantile * Ms_quantile, axis=0)
    
    scaling = np.std(Mu, axis=0) / np.std(Ms, axis=0)
    
    # switch back to non quantile
    if r2_adjust:
        Ms, Mu = Ms, Mu

    # non quantile slope
    gamma_ref = np.sum(Mu * Ms, axis=0) / np.sum(Ms * Ms, axis=0)
    residual_scale = Mu - gamma_ref * Ms
    
    r2 = R2(residual_scale, total=Mu - np.mean(Mu, axis=0))
    
    # select genes
    velocity_genes = np.ones(n_var)
    velocity_genes = (
        (r2 > min_r2)
        & (r2 < max_r2)
        & (gamma_quantile > min_ratio)
        & (gamma_ref > min_ratio)
        & (np.max(Ms > 0, axis=0) > 0)
        & (np.max(Mu > 0, axis=0) > 0)
    )

    # filter noisy genes
    if r2_adjust:
        
        lb, ub = np.nanpercentile(scaling, [10, 90])
        velocity_genes = (
            velocity_genes
            & (scaling > np.min([lb, 0.03]))
            & (scaling < np.max([ub, 3]))
        )
    
    nonzero_s, nonzero_u = Ms > 0, Mu > 0
    weights = np.array(nonzero_s & nonzero_u, dtype=bool)
    nobs = np.sum(weights, axis=0)
        
    velocity_genes = velocity_genes & (nobs > 0.05 * Ms.shape[1])
    
    if inplace:
        adata_raw.var['velocity_genes'] = velocity_genes
    else:
        return adata.var['velocity_genes']
