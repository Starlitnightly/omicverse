r"""PLS-DA and OPLS-DA for multivariate metabolomics analysis.

The discriminant-analysis standard in metabolomics. OPLS-DA (Trygg &
Wold 2002) is preferred over plain PLS-DA because it splits variance
into a single predictive component and one or more orthogonal
components, so the loadings are more interpretable — the S-plot and
VIP scores from OPLS-DA are what users actually publish.

PLS-DA is the sklearn wrapper; OPLS-DA is a custom numpy implementation
(sklearn doesn't ship OPLS). Both return VIP scores for feature
selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function


@dataclass
class PLSDAResult:
    """Output of ``plsda`` / ``opls_da``.

    Attributes
    ----------
    scores : (n_samples, n_components) predictive component scores
    loadings : (n_features, n_components) loadings
    vip : (n_features,) Variable Importance in Projection
    r2x, r2y, q2 : standard PLS-DA quality metrics
    coef : (n_features,) regression coefficient vs the binary Y (for plotting)
    x_ortho_scores : (n_samples, n_ortho) orthogonal scores — only for OPLS-DA
    x_ortho_loadings : (n_features, n_ortho) orthogonal loadings
    group_labels : labels of the two groups in the order used for Y (+1 / -1)
    """

    scores: np.ndarray
    loadings: np.ndarray
    vip: np.ndarray
    r2x: float
    r2y: float
    q2: float
    coef: np.ndarray
    x_ortho_scores: np.ndarray
    x_ortho_loadings: np.ndarray
    group_labels: tuple[str, str]

    def to_vip_table(self, var_names) -> pd.DataFrame:
        """Return a DataFrame of VIP scores sorted descending."""
        return (
            pd.DataFrame({"vip": self.vip, "coef": self.coef}, index=var_names)
            .sort_values("vip", ascending=False)
        )


@register_function(
    aliases=[
        'plsda',
        'PLSDA',
        '偏最小二乘判别',
    ],
    category='metabolomics',
    description='PLS-DA (Partial Least Squares Discriminant Analysis) with VIP scores and leave-one-out Q².',
    examples=[
        "ov.metabol.plsda(adata, group_col='group', n_components=2)",
    ],
    related=[
        'metabol.opls_da',
        'metabol.vip_bar',
    ],
)
def plsda(
    adata: AnnData,
    *,
    group_col: str = "group",
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    n_components: int = 2,
    scale: bool = False,
) -> PLSDAResult:
    """Partial Least Squares Discriminant Analysis (wraps sklearn PLS).

    Parameters
    ----------
    n_components
        Number of latent components. 2 is standard for visualization;
        use leave-one-out Q² to pick the optimal count for
        classification.
    scale
        Scale features inside sklearn PLS (z-score). Usually False here
        because you've already Pareto-scaled via ``transform()``.
    """
    from sklearn.cross_decomposition import PLSRegression

    X, y, labels = _prepare_xy(adata, group_col, group_a, group_b)
    pls = PLSRegression(n_components=n_components, scale=scale)
    pls.fit(X, y)
    scores = pls.x_scores_
    loadings = pls.x_loadings_
    coef = pls.coef_.ravel()
    vip = _vip_from_pls(pls, X)
    r2x = _r2x(X, pls)
    r2y = float(1 - np.sum((y - pls.predict(X).ravel()) ** 2) / np.sum((y - y.mean()) ** 2))
    q2 = _q2_loo(X, y, n_components=n_components, scale=scale)

    return PLSDAResult(
        scores=scores, loadings=loadings, vip=vip, r2x=r2x, r2y=r2y, q2=q2,
        coef=coef,
        x_ortho_scores=np.empty((X.shape[0], 0)),
        x_ortho_loadings=np.empty((X.shape[1], 0)),
        group_labels=labels,
    )


@register_function(
    aliases=[
        'opls_da',
        'OPLS-DA',
        '正交偏最小二乘',
    ],
    category='metabolomics',
    description='OPLS-DA (Trygg & Wold 2002) — single predictive component + orthogonal components for interpretable biomarker discovery.',
    examples=[
        "ov.metabol.opls_da(adata, group_col='group', n_ortho=1)",
    ],
    related=[
        'metabol.s_plot',
        'metabol.vip_bar',
        'metabol.plsda',
    ],
)
def opls_da(
    adata: AnnData,
    *,
    group_col: str = "group",
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    n_ortho: int = 1,
    scale: bool = False,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> PLSDAResult:
    """Orthogonal Projection to Latent Structures — Discriminant Analysis.

    Port of Trygg & Wold 2002 — NIPALS-style iteration that separates
    one predictive component (captures variance correlated with Y)
    from ``n_ortho`` orthogonal components (variance uncorrelated with
    Y but present in X). The predictive component's loadings + VIP are
    the publication-quality output.

    Parameters
    ----------
    n_ortho
        Number of orthogonal components to extract. 1 is usually
        enough for a two-group comparison; increase if the data have
        strong batch or technical structure.
    """
    X, y, labels = _prepare_xy(adata, group_col, group_a, group_b)
    # Mean-center both X and Y (NIPALS convention). Scaling is handled upstream.
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()
    X_work = X_c.copy()
    y_work = y_c.copy()

    T_ortho = np.zeros((X.shape[0], n_ortho))
    P_ortho = np.zeros((X.shape[1], n_ortho))
    for o in range(n_ortho):
        # Predictive weights w from X'y
        w = X_work.T @ y_work
        w = w / np.linalg.norm(w)
        # Predictive scores t = Xw
        t = X_work @ w
        # Predictive loadings p
        p = X_work.T @ t / (t @ t)
        # Orthogonal weights: w_ortho = p - ((w'p)/(w'w)) w
        w_ortho = p - (w @ p / (w @ w)) * w
        w_ortho = w_ortho / np.linalg.norm(w_ortho)
        t_ortho = X_work @ w_ortho
        p_ortho = X_work.T @ t_ortho / (t_ortho @ t_ortho)
        # Deflate X by orthogonal component only (preserve predictive variance)
        X_work = X_work - np.outer(t_ortho, p_ortho)
        T_ortho[:, o] = t_ortho
        P_ortho[:, o] = p_ortho

    # Fit the single predictive component on the orthogonal-deflated X
    w = X_work.T @ y_work
    w = w / np.linalg.norm(w)
    t = X_work @ w
    p = X_work.T @ t / (t @ t)
    # Inner regression coefficient b
    b = (t @ y_work) / (t @ t)

    # VIP for OPLS-DA uses the predictive component's weights.
    vip = _vip_from_weights(w[:, None], p[:, None], np.array([b]), X_c)

    y_pred = b * t
    r2x = float(np.sum((t[:, None] * p[None, :]) ** 2) / np.sum(X_c ** 2))
    r2y = float(1 - np.sum((y_work - y_pred) ** 2) / np.sum(y_work ** 2))
    q2 = _q2_loo_opls(X_c, y_c, n_ortho=n_ortho)

    scores = t.reshape(-1, 1)
    loadings = p.reshape(-1, 1)
    coef = b * w  # back-projected regression coefficient per feature

    return PLSDAResult(
        scores=scores, loadings=loadings, vip=vip,
        r2x=r2x, r2y=r2y, q2=q2,
        coef=coef,
        x_ortho_scores=T_ortho, x_ortho_loadings=P_ortho,
        group_labels=labels,
    )


# ----------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------
def _prepare_xy(adata, group_col, group_a, group_b):
    if group_col not in adata.obs.columns:
        raise KeyError(f"adata.obs has no column {group_col!r}")
    groups = adata.obs[group_col].astype(str).to_numpy()
    unique = pd.unique(groups).tolist()
    if len(unique) < 2:
        raise ValueError(f"{group_col!r} has <2 unique values: {unique}")
    ga = group_a or unique[0]
    gb = group_b or unique[1]
    mask = np.isin(groups, [ga, gb])
    sub = groups[mask]
    y = np.where(sub == ga, +1.0, -1.0)
    X = np.asarray(adata.X[mask], dtype=np.float64)
    # Drop any sample with all-NaN
    finite_rows = np.isfinite(X).all(axis=1)
    return X[finite_rows], y[finite_rows], (ga, gb)


def _vip_from_pls(pls, X):
    """VIP scores from a fitted sklearn PLSRegression.

    VIP_j = sqrt( K * Σ_a (q_a^2 * w_ja^2) / Σ_a q_a^2 )
    where q_a = Y-loading and w_ja = normalized weight.
    """
    w = pls.x_weights_
    q = pls.y_loadings_.ravel()
    t = pls.x_scores_
    # SSY per component (via y_scores) — proportional to q^2 * (t't)
    ss_y = (t ** 2).sum(axis=0) * q ** 2
    total = ss_y.sum() if ss_y.sum() > 0 else 1.0
    weights_sq = (w / np.linalg.norm(w, axis=0)[None, :]) ** 2
    K = X.shape[1]
    vip = np.sqrt(K * (weights_sq @ ss_y) / total)
    return vip


def _vip_from_weights(W, P, b, X):
    """VIP when we have the weights/loadings directly (OPLS-DA path)."""
    # Single predictive component VIP reduces to sqrt(K) * |w_j|
    # but the general form keeps the ss_y-weighted average.
    t = X @ W
    ss_y = (t ** 2).sum(axis=0) * (b ** 2)
    total = ss_y.sum() if ss_y.sum() > 0 else 1.0
    weights_sq = W ** 2
    K = X.shape[1]
    return np.sqrt(K * (weights_sq @ ss_y) / total)


def _r2x(X, pls):
    """Fraction of X-variance explained by all predictive components."""
    t = pls.x_scores_
    p = pls.x_loadings_
    X_c = X - X.mean(axis=0)
    explained = np.sum((t @ p.T) ** 2)
    total = np.sum(X_c ** 2)
    return float(explained / total) if total > 0 else 0.0


def _q2_loo(X, y, *, n_components, scale):
    """Leave-one-out Q² for PLS-DA."""
    from sklearn.cross_decomposition import PLSRegression

    n = X.shape[0]
    press = 0.0
    tss = np.sum((y - y.mean()) ** 2)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        try:
            pls = PLSRegression(n_components=min(n_components, X.shape[1], mask.sum() - 1),
                                scale=scale).fit(X[mask], y[mask])
            pred = pls.predict(X[[i]]).ravel()[0]
            press += (y[i] - pred) ** 2
        except (np.linalg.LinAlgError, ValueError) as exc:
            # Rank-deficient or degenerate fold — warn so users can see why
            # Q² is undefined instead of getting a silent NaN.
            import warnings
            warnings.warn(
                f"PLS-DA Q² LOO fold {i} failed ({type(exc).__name__}: {exc}); "
                "returning NaN. Usually caused by a rank-deficient X after "
                "excluding one sample.",
                UserWarning, stacklevel=2,
            )
            return np.nan
    return float(1 - press / tss) if tss > 0 else 0.0


def _q2_loo_opls(X, y, *, n_ortho):
    """Leave-one-out Q² for OPLS-DA (only the predictive component)."""
    n = X.shape[0]
    press = 0.0
    tss = np.sum(y ** 2)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_tr = X[mask] - X[mask].mean(axis=0)
        y_tr = y[mask] - y[mask].mean()
        x_te = X[[i]] - X[mask].mean(axis=0)
        # Re-run a stripped OPLS-DA
        X_work = X_tr.copy()
        for _ in range(n_ortho):
            w = X_work.T @ y_tr; w = w / np.linalg.norm(w)
            t = X_work @ w
            p = X_work.T @ t / (t @ t)
            w_ortho = p - (w @ p / (w @ w)) * w
            w_ortho = w_ortho / np.linalg.norm(w_ortho)
            t_ortho = X_work @ w_ortho
            p_ortho = X_work.T @ t_ortho / (t_ortho @ t_ortho)
            X_work = X_work - np.outer(t_ortho, p_ortho)
            x_te = x_te - (x_te @ w_ortho) * p_ortho[None, :]
        w = X_work.T @ y_tr; w = w / np.linalg.norm(w)
        t_tr = X_work @ w
        b = (t_tr @ y_tr) / (t_tr @ t_tr)
        t_te = (x_te @ w).ravel()[0]
        pred = b * t_te
        press += (y[i] - y[mask].mean() - pred) ** 2
    return float(1 - press / tss) if tss > 0 else 0.0
