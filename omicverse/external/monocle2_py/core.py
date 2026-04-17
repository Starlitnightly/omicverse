"""
Core data model and preprocessing functions for monocle2_py.

Operates on AnnData objects. Monocle2 metadata is stored in adata.uns['monocle'].
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import gmean


def _init_monocle_uns(adata):
    """Initialize monocle namespace in adata.uns if not present."""
    if 'monocle' not in adata.uns:
        adata.uns['monocle'] = {}


def set_ordering_filter(adata, ordering_genes):
    """
    Mark genes to be used for ordering (trajectory inference).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    ordering_genes : list of str
        Gene names to use for ordering.
    """
    _init_monocle_uns(adata)
    adata.var['use_for_ordering'] = adata.var_names.isin(ordering_genes)
    return adata


def detect_genes(adata, min_expr=0.1):
    """
    Detect genes expressed above a threshold.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    min_expr : float
        Minimum expression threshold.

    Returns
    -------
    adata with updated var['num_cells_expressed'] and obs['num_genes_expressed']
    """
    X = adata.X
    if sparse.issparse(X):
        num_cells = np.array((X > min_expr).sum(axis=0)).flatten()
        num_genes = np.array((X > min_expr).sum(axis=1)).flatten()
    else:
        num_cells = (X > min_expr).sum(axis=0)
        num_genes = (X > min_expr).sum(axis=1)

    adata.var['num_cells_expressed'] = num_cells
    adata.obs['num_genes_expressed'] = num_genes
    return adata


def estimate_size_factors(adata, method='mean-geometric-mean-total',
                          round_exprs=True):
    """
    Estimate size factors matching Monocle2's default method.

    Monocle2 default: method='mean-geometric-mean-total'
      sf = cell_total / exp(mean(log(cell_total)))

    Parameters
    ----------
    adata : AnnData
    method : str
        One of 'mean-geometric-mean-total' (default, matches R Monocle2),
        'median-geometric-mean' (DESeq-style).
    round_exprs : bool
        Round expression values before computing.
    """
    _init_monocle_uns(adata)
    X = adata.X
    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X, dtype=np.float64)

    CM = X_dense.copy()
    if round_exprs:
        CM = np.round(CM)

    if method == 'mean-geometric-mean-total':
        # R: cell_total <- apply(CM, 2, sum)
        #    sfs <- cell_total / exp(mean(log(cell_total)))
        # In R, CM is genes x cells. In Python adata, X is cells x genes.
        cell_total = CM.sum(axis=1)  # sum over genes for each cell
        cell_total[cell_total == 0] = 1  # avoid log(0)
        sfs = cell_total / np.exp(np.mean(np.log(cell_total)))

    elif method == 'median-geometric-mean':
        # DESeq-style
        log_geo_means = np.mean(np.log(CM + 1e-300), axis=0)  # per gene
        sfs = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            norm_cnts = np.log(CM[i, :] + 1e-300) - log_geo_means
            valid = np.isfinite(norm_cnts)
            sfs[i] = np.exp(np.median(norm_cnts[valid])) if valid.sum() > 0 else 1.0

    elif method == 'geometric-mean-total':
        cell_total = CM.sum(axis=1)
        cell_total[cell_total == 0] = 1
        sfs = np.log(cell_total) / np.mean(np.log(cell_total))

    else:
        raise ValueError(f"Unknown method: {method}")

    sfs[np.isnan(sfs)] = 1.0
    adata.obs['Size_Factor'] = sfs
    return adata


def estimate_dispersions(adata, min_cells_detected=1, verbose=False):
    """
    Estimate gene dispersions for negative binomial model.

    Parameters
    ----------
    adata : AnnData
    min_cells_detected : int
        Minimum cells where gene must be detected.
    verbose : bool
    """
    _init_monocle_uns(adata)
    X = adata.X
    if sparse.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X)

    rounded = np.round(X_dense)

    # Filter genes by detection
    n_detected = (rounded > 0).sum(axis=0)
    valid_mask = n_detected > min_cells_detected

    if 'Size_Factor' not in adata.obs.columns:
        estimate_size_factors(adata)

    sf = adata.obs['Size_Factor'].values

    # Match R's disp_calc_helper_NB exactly:
    # x <- t(t(rounded) / Size_Factor)  (normalize by size factor)
    # xim <- mean(1/Size_Factor)
    # f_expression_mean <- rowMeans(x)
    # f_expression_var <- rowMeans((x - f_expression_mean)^2)
    # disp = (var - xim * mean) / mean^2
    normed = rounded / sf[:, None]  # cells x genes, each cell divided by its SF
    xim = np.mean(1.0 / sf)

    # Gene-wise mean and variance (across cells)
    mu = normed.mean(axis=0)  # rowMeans in R (genes are rows there)
    f_var = np.mean((normed - mu[None, :]) ** 2, axis=0)  # rowMeans((x - mean)^2)

    # Method of moments dispersion: (var - xim*mean) / mean^2
    disp = np.full(adata.n_vars, np.nan)
    valid = (mu > 0) & valid_mask
    disp[valid] = (f_var[valid] - xim * mu[valid]) / (mu[valid] ** 2)
    disp[disp < 0] = 0  # R sets negative dispersions to 0

    adata.var['mean_expression'] = mu
    adata.var['dispersion_fit'] = disp
    adata.var['dispersion_empirical'] = disp

    # Parametric fit matching R's parametricDispersionFit exactly:
    # R code:
    #   fit <- glm(disp ~ I(1/mu), data=good, family=Gamma(link="identity"), start=coefs)
    #   Iterate: filter residuals, refit, until convergence
    valid_for_fit = np.isfinite(disp) & (mu > 0) & (disp > 0)
    if valid_for_fit.sum() > 10:
        try:
            import statsmodels.api as sm
            import warnings as _warnings

            mu_all = mu[valid_for_fit]
            disp_all = disp[valid_for_fit]

            def _parametric_dispersion_fit(mu_v, disp_v, start_coefs):
                """Match R's parametricDispersionFit exactly."""
                coefs = start_coefs.copy()
                fit_result = None  # explicit sentinel — avoids 'in dir()' scoping bugs
                for _iter in range(12):
                    pred = coefs[0] + coefs[1] / mu_v
                    pred[pred <= 0] = 1e-10
                    residuals = disp_v / pred
                    keep = (residuals > 1e-6) & (residuals < 10000)
                    if keep.sum() < 3:
                        break

                    X_design = np.column_stack([np.ones(keep.sum()),
                                                1.0 / mu_v[keep]])
                    y = disp_v[keep]

                    try:
                        with _warnings.catch_warnings():
                            _warnings.simplefilter("ignore")
                            glm_model = sm.GLM(
                                y, X_design,
                                family=sm.families.Gamma(
                                    sm.families.links.Identity()))
                            fit_result = glm_model.fit(start_params=coefs)
                            new_coefs = fit_result.params.copy()
                    except Exception:
                        break

                    old_coefs = coefs.copy()
                    coefs = new_coefs
                    if coefs[0] < 1e-6:
                        coefs[0] = 1e-6
                    if coefs[1] < 0:
                        break
                    if np.sum(np.log(coefs / old_coefs) ** 2) < 1e-6:
                        break
                return coefs, fit_result

            # First fit
            coefs, fit_result = _parametric_dispersion_fit(
                mu_all, disp_all, np.array([1e-6, 1.0]))

            # Outlier removal (matching R's removeOutliers=TRUE).
            # NOTE on index alignment: `fit_result` was obtained on
            # `X_design = mu_all[keep]` (see _parametric_dispersion_fit).
            # Cook's distance therefore has length `keep.sum()` and is
            # aligned with `keep_idx` — NOT with `mu_all`.
            if fit_result is not None:
                try:
                    # Recompute the same `keep` mask used in the final
                    # _parametric_dispersion_fit iteration
                    pred = coefs[0] + coefs[1] / mu_all
                    pred[pred <= 0] = 1e-10
                    residuals = disp_all / pred
                    keep = (residuals > 1e-6) & (residuals < 10000)
                    keep_idx = np.where(keep)[0]

                    influence = fit_result.get_influence()
                    cooks_d = influence.cooks_distance[0]
                    cooks_cutoff = 4.0 / len(mu_all)

                    # `cooks_d` has one entry per row of the last GLM fit.
                    # That fit used rows `keep_idx`, so they share the
                    # same index space — no slicing / truncation needed.
                    if len(cooks_d) != len(keep_idx):
                        # Statsmodels may have dropped some rows internally
                        # (e.g. zero-weight observations). Fall back to a
                        # trailing alignment rather than silently mis-indexing.
                        m = min(len(cooks_d), len(keep_idx))
                        cooks_d_aligned = cooks_d[:m]
                        keep_idx_aligned = keep_idx[:m]
                    else:
                        cooks_d_aligned = cooks_d
                        keep_idx_aligned = keep_idx

                    outlier_in_keep = cooks_d_aligned > cooks_cutoff
                    outlier_genes = set(
                        keep_idx_aligned[outlier_in_keep].tolist()
                    )
                    n_outliers = len(outlier_genes)
                    if verbose:
                        print(f"  Removing {n_outliers} outliers")

                    if n_outliers > 0 and n_outliers < len(mu_all) - 10:
                        clean_mask = np.array([
                            i not in outlier_genes for i in range(len(mu_all))
                        ])
                        coefs, _ = _parametric_dispersion_fit(
                            mu_all[clean_mask], disp_all[clean_mask],
                            coefs,
                        )
                except Exception as _e:
                    # If Cook's distance / re-fit fails, keep the first
                    # fit and emit a warning so the user knows.
                    _warnings.warn(
                        f"Outlier removal step failed: {_e!r}. "
                        "Keeping initial dispersion fit.",
                        RuntimeWarning,
                    )

            if coefs[0] > 0 and coefs[1] > 0:
                fitted_disp = coefs[0] + coefs[1] / mu
                fitted_disp[mu <= 0] = np.nan
                adata.var['dispersion_fit'] = fitted_disp

                def disp_func(x, _c=coefs.copy()):
                    return _c[0] + _c[1] / x

                adata.uns['monocle']['disp_func'] = disp_func
                if verbose:
                    print(f"  Dispersion fit coefs: asymptDisp={coefs[0]:.6f}, "
                          f"extraPois={coefs[1]:.5f}")
            else:
                # Fit converged to degenerate coefficients — warn so the
                # caller can tell that downstream NB-GLM fits won't get
                # a disp_func hint.
                _warnings.warn(
                    f"Parametric dispersion fit produced invalid "
                    f"coefficients (a={coefs[0]:.4g}, b={coefs[1]:.4g}); "
                    "no disp_func will be stored.",
                    RuntimeWarning,
                )
        except Exception as _exc:
            # Fitting framework itself failed — keep the empirical
            # dispersions but warn loudly. Downstream GLM fits will
            # fall back to their default behaviour.
            _warnings.warn(
                f"Dispersion GLM fit failed ({_exc!r}); keeping "
                "empirical dispersions only. differential_gene_test "
                "and BEAM will still work but without a dispersion "
                "hint.",
                RuntimeWarning,
            )

    return adata


def estimate_t(relative_expr_matrix, relative_expr_thresh=0.1):
    """Estimate the detection threshold t for Census normalization.

    Matches Monocle2's `estimate_t`: per cell, take the mode of log10(expr)
    values above the threshold and return 10^mode.
    """
    X = relative_expr_matrix
    if sparse.issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=float)  # cells x genes

    from scipy.stats import gaussian_kde
    t_values = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        vals = X[i][X[i] > relative_expr_thresh]
        if len(vals) < 2:
            t_values[i] = 1.0
            continue
        log_vals = np.log10(vals)
        try:
            kde = gaussian_kde(log_vals)
            grid = np.linspace(log_vals.min(), log_vals.max(), 1024)
            density = kde(grid)
            mode_log = grid[np.argmax(density)]
            t_values[i] = 10 ** mode_log
        except Exception:
            t_values[i] = np.median(vals)
    return t_values


def relative2abs(adata, t_estimate=None, method='num_genes',
                 expected_capture_rate=0.25,
                 return_all=False, verbose=False):
    """Census normalization: convert relative expression (TPM/FPKM) to
    estimated absolute transcript counts.

    Simplified implementation of Monocle2's `relative2abs`.

    Parameters
    ----------
    adata : AnnData
    t_estimate : array or None
        Detection threshold per cell; computed via `estimate_t` if None.
    method : 'num_genes' (default)
    expected_capture_rate : float
        Approximate fraction of transcripts captured.

    Returns
    -------
    AnnData with X replaced by estimated absolute counts.
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.array(X, dtype=float)  # cells x genes

    if t_estimate is None:
        t_estimate = estimate_t(X)

    # Simple Census model: scale each cell so that the mode of expressed
    # genes corresponds to ~1 transcript.
    # num_genes method: total transcripts ≈ num_detected_genes / capture_rate
    n_detected = (X > 0.1).sum(axis=1)
    total_transcripts = n_detected / expected_capture_rate

    # Rescale each cell so the total abundance matches estimate
    row_sums = X.sum(axis=1)
    row_sums[row_sums == 0] = 1
    scale = total_transcripts / row_sums
    X_abs = X * scale[:, None]
    X_abs = np.round(X_abs)

    new_adata = adata.copy()
    new_adata.X = X_abs.astype(np.float64)
    if return_all:
        return {
            'norm_cds': new_adata,
            't_estimate': t_estimate,
            'total_transcripts': total_transcripts,
        }
    return new_adata


def dispersion_table(adata):
    """
    Return dispersion table as a DataFrame.

    Parameters
    ----------
    adata : AnnData

    Returns
    -------
    pd.DataFrame with columns: gene_id, mean_expression, dispersion_fit, dispersion_empirical
    """
    df = pd.DataFrame({
        'gene_id': adata.var_names,
        'mean_expression': adata.var.get('mean_expression', np.nan),
        'dispersion_fit': adata.var.get('dispersion_fit', np.nan),
        'dispersion_empirical': adata.var.get('dispersion_empirical', np.nan),
    })
    df.index = adata.var_names
    return df
