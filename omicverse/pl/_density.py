import numpy as np
import scanpy as sc
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt



def calculate_gene_density(
    adata,
    features,
    basis="X_umap",
    dims=(0, 1),
    adjust=1,
    min_expr=0.1,          # NEW: minimal raw expression to keep as weight > 0
):
    """
    Weighted KDE on a 2-D embedding with min–max scaled weights.
    Cells whose raw expression < `min_expr` are excluded from the KDE fit.

    Compute a weighted kernel density estimate (KDE) for each feature
    and store the per-cell density values in `adata.obs`.

    Parameters
    ----------
    adata : AnnData
        AnnData object that contains the embedding in `adata.obsm[basis]`.
    features : list[str]
        Feature names (gene names or pre-computed scores) to process.
    basis : str, default "X_umap"
        Key in `adata.obsm` that stores the 2-D embedding (e.g., UMAP).
    dims : tuple[int, int], default (0, 1)
        Indices of the two embedding dimensions to use.
    adjust : float, default 1
        Bandwidth scaling factor passed to `scipy.stats.gaussian_kde`.

    """
    if len(dims) != 2:
        raise ValueError("`dims` must have length 2")
    if basis not in adata.obsm:
        raise ValueError(f"Embedding '{basis}' not found.")

    emb_all = adata.obsm[basis][:, dims]          # (n_cells, 2)

    for feat in features:
        # ----- fetch raw weights -------------------------------------------
        if feat in adata.obs:
            w_raw = adata.obs[feat].to_numpy()
        elif feat in adata.var_names:
            w_raw = adata[:, feat].X.toarray().ravel()
        else:
            raise ValueError(f"Feature '{feat}' not found in obs or var.")

        # ----- validity mask: finite coords & finite expr -------------------
        mask_finite = np.isfinite(w_raw) & np.all(np.isfinite(emb_all), axis=1)

        # ----- NEW: expression threshold -----------------------------------
        mask_expr   = w_raw > min_expr
        mask_train  = mask_finite & mask_expr

        emb_train   = emb_all[mask_train]
        w_train_raw = w_raw[mask_train]

        if emb_train.shape[0] < 5:
            print(f"[{feat}] too few cells above threshold; skipping KDE.")
            adata.obs[f"density_{feat}"] = np.nan
            continue

        # ----- min–max scale to 0-1 ----------------------------------------
        w_min, w_max = w_train_raw.min(), w_train_raw.max()
        w_train      = (w_train_raw - w_min) / (w_max - w_min)

        # ----- KDE fit ------------------------------------------------------
        kde = gaussian_kde(emb_train.T, weights=w_train, bw_method=adjust)

        # ----- evaluate on ALL cells ---------------------------------------
        density = kde(emb_all.T)
        adata.obs[f"density_{feat}"] = density

        print(f"✅ density_{feat} written (train cells = {emb_train.shape[0]})")


import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

def add_density_contour(
    ax,
    embeddings,              # (n_cells, 2) array
    weights,                 # 1-D array, will be min-max scaled
    levels="quantile",       # "quantile" or a numeric list, see below
    n_quantiles=5,
    bw_adjust=0.3,
    cmap_contour="Greys",
    linewidth=1.0,
    zorder=10,
    fill=False,
    alpha=0.4,
):
    """
    Draw KDE iso-density contours on an existing matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes that already hosts your scatter / scanpy embedding.
    embeddings : ndarray, shape (n, 2)
        2-D coordinates (e.g. UMAP).
    weights : ndarray, shape (n,)
        Raw expression or any weight; will be min-max scaled to [0, 1].
    levels : str | list[float]
        * "quantile": use `n_quantiles` equally spaced quantiles (e.g. 0.2,0.4,…)
        * list/tuple  : explicit contour levels.
    n_quantiles : int
        Number of quantile levels when `levels="quantile"`.
    bw_adjust : float
        Bandwidth factor for gaussian_kde (smaller = sharper contours).
    fill : bool
        True → use `ax.contourf` (filled), False → `ax.contour` (lines).
    alpha : float
        Transparency for filled contours.
    """
    # ---------- fit KDE ----------------------------------------------------
    w_min, w_max = weights.min(), weights.max()
    w_norm = None if w_max == w_min else (weights - w_min) / (w_max - w_min)
    kde = gaussian_kde(embeddings.T, weights=w_norm, bw_method=bw_adjust)

    # ---------- prepare evaluation grid -----------------------------------
    xmin, xmax = embeddings[:, 0].min(), embeddings[:, 0].max()
    ymin, ymax = embeddings[:, 1].min(), embeddings[:, 1].max()
    xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]   # 300×300 grid
    grid = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # ---------- determine contour levels ----------------------------------
    if levels == "quantile":
        qs = np.linspace(0, 1, n_quantiles + 2)[1:-1]   # drop 0 & 1
        levels = np.quantile(grid, qs)

    # ---------- draw -------------------------------------------------------
    if fill:
        cs = ax.contourf(
            xx, yy, grid,
            levels=levels,
            cmap=cmap_contour,
            alpha=alpha,
            zorder=zorder,
        )
    else:
        cs = ax.contour(
            xx, yy, grid,
            levels=levels,
            cmap=cmap_contour,
            linewidths=linewidth,
            zorder=zorder,
        )
    return cs   # so you can add a colorbar if desired

