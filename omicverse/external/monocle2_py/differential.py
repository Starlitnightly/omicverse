"""
Differential expression analysis for monocle2_py.

Implements differentialGeneTest, BEAM, fitModel, genSmoothCurves.
Uses GLM with natural splines as a pure Python replacement for VGAM.
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.interpolate import BSpline
import warnings


def _natural_spline_basis(x, df=3):
    """
    Generate natural cubic spline basis matrix.

    Parameters
    ----------
    x : np.ndarray
        Predictor values.
    df : int
        Degrees of freedom (number of basis functions).

    Returns
    -------
    basis : np.ndarray, shape (len(x), df)
    """
    from scipy.interpolate import splrep, BSpline as BSplineScipy

    x = np.asarray(x, dtype=float)
    n = len(x)

    if df < 1:
        df = 1

    # Use quantile-based knots
    n_inner_knots = df - 1
    if n_inner_knots > 0:
        quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
        knots = np.quantile(x, quantiles)
    else:
        knots = np.array([])

    x_min, x_max = x.min(), x.max()

    # Build B-spline basis of degree 3
    all_knots = np.concatenate([
        [x_min] * 4,
        knots,
        [x_max] * 4,
    ])

    n_basis = len(all_knots) - 4
    basis = np.zeros((n, n_basis))

    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spl = BSplineScipy(all_knots, coeffs, 3)
        basis[:, i] = spl(x)

    # Apply natural spline constraints (linear beyond boundaries)
    # Simple approach: just return the first df columns
    if basis.shape[1] > df:
        # Use QR decomposition to get orthogonal basis
        Q, R = np.linalg.qr(basis)
        basis = Q[:, :df]
    elif basis.shape[1] < df:
        df = basis.shape[1]

    return basis


def _fit_glm_nb(y, X, size_factors=None, max_iter=25):
    """
    Fit a negative binomial GLM (log link) — vectorized IRLS without diag matrices.
    Uses X^T @ (w[:, None] * X) instead of X.T @ diag(w) @ X for speed.
    """
    n = len(y)
    p = X.shape[1]

    if size_factors is not None:
        offset = np.log(size_factors)
    else:
        offset = np.zeros(n)

    y = np.asarray(y, dtype=float)

    # Initialize with log(y+0.5)
    eta = np.log(y + 0.5) - offset
    try:
        beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(p)

    # Estimate theta via method of moments
    var_y = np.var(y, ddof=1)
    mean_y = np.mean(y)
    if var_y > mean_y and mean_y > 0:
        theta = mean_y ** 2 / (var_y - mean_y)
        theta = np.clip(theta, 0.01, 1e6)
    else:
        theta = 1e6  # Poisson-like

    converged = False
    for _iter in range(max_iter):
        eta = X @ beta + offset
        # Clip eta to avoid overflow in exp
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-10)

        # NB working weights and response (identity link trick)
        w = mu / (1.0 + mu / theta)
        z = eta - offset + (y - mu) / np.maximum(w, 1e-10)

        # Weighted least squares via X^T @ (w[:,None] * X) — no dense diag!
        Xw = X * w[:, None]
        try:
            XtWX = X.T @ Xw + 1e-8 * np.eye(p)
            XtWz = Xw.T @ z
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < 1e-6:
            converged = True
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta + offset
    eta = np.clip(eta, -30, 30)
    mu = np.exp(eta)
    mu = np.maximum(mu, 1e-10)

    # Log-likelihood
    y_int = np.round(y).astype(np.int64).clip(min=0)
    p_nb = theta / (theta + mu)
    loglik = np.sum(stats.nbinom.logpmf(y_int, n=theta, p=p_nb))

    return {
        'coefficients': beta,
        'loglik': loglik,
        'theta': theta,
        'fitted': mu,
        'converged': converged,
    }


def _fit_glm_gaussian(y, X):
    """Fit a Gaussian GLM (identity link)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    mu = X @ beta
    residuals = y - mu
    n = len(y)
    p = X.shape[1]
    sigma2 = np.sum(residuals ** 2) / max(n - p, 1)
    loglik = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals ** 2) / (2 * sigma2)
    return {
        'coefficients': beta,
        'loglik': loglik,
        'fitted': mu,
        'converged': True,
    }


def _build_design_matrix(pseudotime, df=3, branch=None):
    """
    Build design matrix with natural spline basis.

    Parameters
    ----------
    pseudotime : np.ndarray
    df : int
    branch : np.ndarray or None

    Returns
    -------
    X_full : np.ndarray - full model design matrix
    X_reduced : np.ndarray - reduced model design matrix
    """
    ns_basis = _natural_spline_basis(pseudotime, df=df)
    intercept = np.ones((len(pseudotime), 1))

    if branch is not None:
        branch_vals = np.unique(branch)
        if len(branch_vals) <= 1:
            X_full = np.hstack([intercept, ns_basis])
            X_reduced = np.hstack([intercept])
        else:
            # Branch indicator
            branch_indicator = (branch == branch_vals[1]).astype(float)[:, None]
            # Interaction terms
            interaction = ns_basis * branch_indicator
            X_full = np.hstack([intercept, ns_basis, branch_indicator, interaction])
            X_reduced = np.hstack([intercept, ns_basis])
    else:
        X_full = np.hstack([intercept, ns_basis])
        X_reduced = np.hstack([intercept])

    return X_full, X_reduced


def _diff_test_categorical(expr, X_full, X_reduced, size_factors,
                            expression_family):
    """LRT with pre-computed design matrices (for categorical covariates)."""
    try:
        if expression_family in ('negbinomial', 'negbinomial.size'):
            y = np.round(expr).astype(float)
            full_fit = _fit_glm_nb(y, X_full, size_factors=size_factors)
            reduced_fit = _fit_glm_nb(y, X_reduced, size_factors=size_factors)
        else:
            y = expr.astype(float)
            full_fit = _fit_glm_gaussian(y, X_full)
            reduced_fit = _fit_glm_gaussian(y, X_reduced)

        lr_stat = max(2 * (full_fit['loglik'] - reduced_fit['loglik']), 0)
        df_diff = X_full.shape[1] - X_reduced.shape[1]
        if df_diff > 0:
            pval = 1 - stats.chi2.cdf(lr_stat, df_diff)
        else:
            pval = 1.0
        return {'status': 'OK', 'pval': pval, 'family': expression_family}
    except Exception:
        return {'status': 'FAIL', 'pval': 1.0, 'family': expression_family}


def _diff_test_single_gene(expr, pseudotime, size_factors, df=3,
                            expression_family='negbinomial',
                            branch=None, relative_expr=True):
    """
    Differential expression test for a single gene.

    Returns dict with status, pval, family.
    """
    try:
        X_full, X_reduced = _build_design_matrix(pseudotime, df=df, branch=branch)

        if expression_family in ('negbinomial', 'negbinomial.size'):
            y = np.round(expr).astype(float)
            if relative_expr and size_factors is not None:
                sf = size_factors
            else:
                sf = None

            full_fit = _fit_glm_nb(y, X_full, size_factors=sf)
            reduced_fit = _fit_glm_nb(y, X_reduced, size_factors=sf)
        else:
            y = expr.astype(float)
            full_fit = _fit_glm_gaussian(y, X_full)
            reduced_fit = _fit_glm_gaussian(y, X_reduced)

        # Likelihood ratio test
        lr_stat = 2 * (full_fit['loglik'] - reduced_fit['loglik'])
        lr_stat = max(lr_stat, 0)
        df_diff = X_full.shape[1] - X_reduced.shape[1]

        if df_diff > 0:
            pval = 1 - stats.chi2.cdf(lr_stat, df_diff)
        else:
            pval = 1.0

        return {'status': 'OK', 'pval': pval, 'family': expression_family}

    except Exception:
        return {'status': 'FAIL', 'pval': 1.0, 'family': expression_family}


def differential_gene_test(adata, fullModelFormulaStr="~sm.ns(Pseudotime, df=3)",
                           reducedModelFormulaStr="~1",
                           relative_expr=True, cores=-1, verbose=False):
    """
    Test each gene for differential expression along pseudotime.

    Parameters
    ----------
    adata : AnnData
    fullModelFormulaStr : str
        Full model formula (parsed for df parameter).
    reducedModelFormulaStr : str
        Reduced model formula.
    relative_expr : bool
        Whether to use size-factor-normalized expression.
    cores : int
        Number of cores (uses joblib if > 1).
    verbose : bool

    Returns
    -------
    pd.DataFrame with columns: status, family, pval, qval, gene_short_name, etc.
    """
    # Parse df from formula
    import re
    df_match = re.search(r'df\s*=\s*(\d+)', fullModelFormulaStr)
    df = int(df_match.group(1)) if df_match else 3

    if 'Size_Factor' in adata.obs.columns:
        size_factors = adata.obs['Size_Factor'].values
    else:
        size_factors = None

    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.array(X)

    # Determine expression family
    is_count = np.all(X == np.round(X))
    expression_family = 'negbinomial' if is_count else 'gaussian'

    # Determine formula type: Pseudotime/Branch vs. categorical (Cluster/Type)
    has_pseudotime = 'Pseudotime' in fullModelFormulaStr
    has_branch = 'Branch' in fullModelFormulaStr

    # Extract main covariate(s) other than Pseudotime/Branch
    # Simple parser for "~Cluster" / "~Type" etc.
    terms = fullModelFormulaStr.replace('~', '').strip()
    categorical_covar = None
    if not has_pseudotime and terms and terms != '1':
        # Use the first term as a categorical covariate
        first_term = re.split(r'[+*:]', terms)[0].strip()
        if first_term in adata.obs.columns:
            categorical_covar = first_term

    # Precompute design matrices when using categorical covariate
    # (same for all genes — huge speedup over per-gene formula parsing)
    precomputed_X_full = None
    precomputed_X_reduced = None
    pseudotime = None
    branch_vals = None

    if categorical_covar is not None:
        # One-hot encode categorical covariate
        cat_vals = adata.obs[categorical_covar].astype('category')
        dummies = pd.get_dummies(cat_vals, drop_first=True).values.astype(float)
        intercept = np.ones((adata.n_obs, 1))
        precomputed_X_full = np.hstack([intercept, dummies])
        precomputed_X_reduced = intercept
    elif has_pseudotime:
        pseudotime = adata.obs['Pseudotime'].values
        if has_branch and 'Branch' in adata.obs.columns:
            branch_vals = adata.obs['Branch'].values
    else:
        # Default: reduced-only model — everything should be NS
        pass

    results = []

    def _test_gene(i):
        expr = X[:, i]
        if precomputed_X_full is not None:
            return _diff_test_categorical(
                expr, precomputed_X_full, precomputed_X_reduced,
                size_factors if relative_expr else None,
                expression_family,
            )
        return _diff_test_single_gene(
            expr, pseudotime, size_factors, df=df,
            expression_family=expression_family,
            branch=branch_vals, relative_expr=relative_expr,
        )

    if cores == -1 or cores > 1:
        try:
            from joblib import Parallel, delayed
            n_jobs = cores if cores > 0 else -1
            if verbose:
                print(f"  Running in parallel with n_jobs={n_jobs}")
            results = Parallel(n_jobs=n_jobs, batch_size=50)(
                delayed(_test_gene)(i) for i in range(adata.n_vars)
            )
        except ImportError:
            results = [_test_gene(i) for i in range(adata.n_vars)]
    else:
        for i in range(adata.n_vars):
            if verbose and i % 500 == 0:
                print(f"  Testing gene {i}/{adata.n_vars}")
            results.append(_test_gene(i))

    df_results = pd.DataFrame(results, index=adata.var_names)

    # BH correction
    ok_mask = df_results['status'] == 'OK'
    df_results['qval'] = 1.0
    if ok_mask.sum() > 0:
        from statsmodels.stats.multitest import multipletests
        _, qvals, _, _ = multipletests(df_results.loc[ok_mask, 'pval'].values, method='fdr_bh')
        df_results.loc[ok_mask, 'qval'] = qvals

    # Merge with gene metadata
    if 'gene_short_name' in adata.var.columns:
        df_results['gene_short_name'] = adata.var['gene_short_name'].values

    return df_results


def fit_model(adata, modelFormulaStr="~sm.ns(Pseudotime, df=3)",
              relative_expr=True, cores=1):
    """
    Fit a model to each gene.

    Returns a list of model fit dictionaries.
    """
    import re
    df_match = re.search(r'df\s*=\s*(\d+)', modelFormulaStr)
    df = int(df_match.group(1)) if df_match else 3

    pseudotime = adata.obs['Pseudotime'].values
    size_factors = adata.obs.get('Size_Factor', pd.Series(np.ones(adata.n_obs))).values

    X_data = adata.X
    if sparse.issparse(X_data):
        X_data = X_data.toarray()
    X_data = np.array(X_data)

    is_count = np.all(X_data == np.round(X_data))

    # Check for Branch
    has_branch = 'Branch' in modelFormulaStr
    branch = adata.obs.get('Branch', None)
    branch_vals = branch.values if has_branch and branch is not None else None

    X_full, _ = _build_design_matrix(pseudotime, df=df, branch=branch_vals)

    models = []
    for i in range(adata.n_vars):
        expr = X_data[:, i]
        try:
            if is_count:
                fit = _fit_glm_nb(np.round(expr), X_full, size_factors=size_factors)
            else:
                fit = _fit_glm_gaussian(expr, X_full)
            fit['gene'] = adata.var_names[i]
            models.append(fit)
        except Exception:
            models.append(None)

    return models


def gen_smooth_curves(adata, new_data=None, trend_formula="~sm.ns(Pseudotime, df=3)",
                      relative_expr=True, cores=1):
    """
    Generate smoothed expression curves for each gene.

    Parameters
    ----------
    adata : AnnData
    new_data : pd.DataFrame or None
        DataFrame with Pseudotime column (and optionally Branch).
        If None, uses the existing pseudotime values.
    trend_formula : str
    relative_expr : bool
    cores : int

    Returns
    -------
    np.ndarray, shape (n_genes, n_points) - smoothed expression
    """
    import re
    df_match = re.search(r'df\s*=\s*(\d+)', trend_formula)
    df = int(df_match.group(1)) if df_match else 3

    pseudotime_train = adata.obs['Pseudotime'].values
    size_factors = adata.obs.get('Size_Factor', pd.Series(np.ones(adata.n_obs))).values

    X_data = adata.X
    if sparse.issparse(X_data):
        X_data = X_data.toarray()
    X_data = np.array(X_data)

    is_count = np.all(X_data == np.round(X_data))

    # Training design matrix
    has_branch = 'Branch' in trend_formula
    branch_train = adata.obs.get('Branch', None)
    branch_train_vals = branch_train.values if has_branch and branch_train is not None else None

    X_train, _ = _build_design_matrix(pseudotime_train, df=df, branch=branch_train_vals)

    # Prediction data
    if new_data is not None:
        pseudotime_pred = new_data['Pseudotime'].values
        branch_pred_vals = new_data.get('Branch', pd.Series([None] * len(new_data))).values
        if has_branch and branch_pred_vals is not None:
            X_pred, _ = _build_design_matrix(pseudotime_pred, df=df, branch=branch_pred_vals)
        else:
            X_pred, _ = _build_design_matrix(pseudotime_pred, df=df)
        n_points = len(new_data)
    else:
        X_pred = X_train
        n_points = len(pseudotime_train)

    expression_curves = np.full((adata.n_vars, n_points), np.nan)

    for i in range(adata.n_vars):
        try:
            expr = X_data[:, i]
            if is_count:
                fit = _fit_glm_nb(np.round(expr), X_train, size_factors=size_factors)
                eta = X_pred @ fit['coefficients']
                expression_curves[i, :] = np.exp(eta)
            else:
                fit = _fit_glm_gaussian(expr, X_train)
                expression_curves[i, :] = X_pred @ fit['coefficients']
        except Exception:
            pass

    return expression_curves


def BEAM(adata, branch_point=1, branch_states=None, branch_labels=None,
         fullModelFormulaStr="~sm.ns(Pseudotime, df=3)*Branch",
         reducedModelFormulaStr="~sm.ns(Pseudotime, df=3)",
         relative_expr=True, verbose=False, cores=1):
    """
    Branch Expression Analysis Modeling.

    Tests each gene for differential expression between branches.

    Parameters
    ----------
    adata : AnnData
    branch_point : int
        Which branch point to test.
    branch_states : list or None
        States defining the branches.
    branch_labels : list or None
    fullModelFormulaStr : str
    reducedModelFormulaStr : str
    relative_expr : bool
    verbose : bool
    cores : int

    Returns
    -------
    pd.DataFrame with test results
    """
    monocle = adata.uns.get('monocle', {})

    # Build branch CDS subset
    if branch_states is None:
        branch_points = monocle.get('branch_points', [])
        if branch_point < 1 or branch_point > len(branch_points):
            raise ValueError(
                f"Branch point index {branch_point} out of range "
                f"(found {len(branch_points)} branch points)."
            )
        # Find the children states specific to THIS branch point by
        # traversing the Y-centre MST: for the selected branch vertex,
        # enumerate its adjacent subtrees, drop the one containing the
        # root (Pseudotime==0) cell, and take the two with highest-mean
        # pseudotime as the competing lineages.
        mst = monocle.get('mst')
        closest_vertex = monocle.get('pr_graph_cell_proj_closest_vertex')
        branch_vertex_name = branch_points[branch_point - 1]

        if mst is None or closest_vertex is None:
            # Fallback: keep previous heuristic if MST metadata missing
            states_all = sorted(adata.obs['State'].unique())
            if (adata.obs['Pseudotime'] == 0).any():
                root_state = adata.obs.loc[
                    adata.obs['Pseudotime'] == adata.obs['Pseudotime'].min(),
                    'State'].iloc[0]
            else:
                root_state = adata.obs['State'].value_counts().idxmax()
            non_root = [s for s in states_all if s != root_state]
            state_pt = {s: adata.obs.loc[adata.obs['State'] == s, 'Pseudotime'].mean()
                        for s in non_root}
            branch_states = sorted(state_pt, key=state_pt.get, reverse=True)[:2]
        else:
            branch_vertex = mst.vs['name'].index(branch_vertex_name)
            # Identify root state (min pseudotime)
            root_state = adata.obs.loc[adata.obs['Pseudotime'].idxmin(), 'State']
            # Remove the branch vertex, find connected components of children
            neighbours = mst.neighbors(branch_vertex)
            g_no_bp = mst.copy()
            g_no_bp.delete_vertices([branch_vertex])
            components = g_no_bp.connected_components()
            # Which Y-vertex is in which component?
            name_to_comp = {v['name']: components.membership[v.index]
                            for v in g_no_bp.vs}
            # Map each cell to a Y-vertex → component
            candidate_state_pt = {}
            for st in adata.obs['State'].unique():
                if st == root_state:
                    continue
                cells = adata.obs.index[adata.obs['State'] == st]
                y_ids = closest_vertex[np.isin(adata.obs_names, cells)]
                y_names = [f'Y_{y}' if isinstance(y, (int, np.integer))
                           else (mst.vs[y]['name'] if y < mst.vcount() else None)
                           for y in np.atleast_1d(y_ids).ravel()]
                # Count which component dominates
                comps = [name_to_comp[n] for n in y_names
                         if n in name_to_comp]
                if not comps:
                    continue
                dominant = max(set(comps), key=comps.count)
                key = dominant
                if key not in candidate_state_pt:
                    candidate_state_pt[key] = (
                        st,
                        adata.obs.loc[adata.obs['State'] == st,
                                      'Pseudotime'].mean(),
                    )
            # Take the two components with highest-mean-pseudotime state
            picked = sorted(candidate_state_pt.values(),
                            key=lambda t: t[1], reverse=True)[:2]
            branch_states = [p[0] for p in picked]

    if len(branch_states) < 2:
        n_states = adata.obs['State'].nunique()
        raise ValueError(
            f"Need at least 2 branch states for BEAM, found {branch_states}. "
            f"Only {n_states} total states in trajectory — increase ncenter "
            f"or use reduce_dimension(..., ncenter=N//8)."
        )

    # Build subset with Branch column
    mask1 = adata.obs['State'].isin([branch_states[0]])
    mask2 = adata.obs['State'].isin([branch_states[1]])

    # Include progenitor cells (earlier pseudotime)
    root_state_mask = ~adata.obs['State'].isin(branch_states)
    combined_mask = mask1 | mask2 | root_state_mask

    adata_subset = adata[combined_mask].copy()

    # Assign Branch labels
    branch_col = np.full(adata_subset.n_obs, 'Pre-branch', dtype=object)
    branch_col[adata_subset.obs['State'].isin([branch_states[0]]).values] = \
        branch_labels[0] if branch_labels else f'State_{branch_states[0]}'
    branch_col[adata_subset.obs['State'].isin([branch_states[1]]).values] = \
        branch_labels[1] if branch_labels else f'State_{branch_states[1]}'
    adata_subset.obs['Branch'] = pd.Categorical(branch_col)

    # Scale pseudotime 0-100
    pt = adata_subset.obs['Pseudotime'].values.copy()
    if pt.max() > pt.min():
        pt = 100 * (pt - pt.min()) / (pt.max() - pt.min())
    adata_subset.obs['Pseudotime'] = pt

    # Run differential gene test with branch
    result = differential_gene_test(
        adata_subset,
        fullModelFormulaStr=fullModelFormulaStr,
        reducedModelFormulaStr=reducedModelFormulaStr,
        relative_expr=relative_expr,
        cores=cores,
        verbose=verbose,
    )

    return result
