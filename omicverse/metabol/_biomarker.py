r"""Biomarker discovery for metabolomics — ROC/AUC + nested-CV panels.

Clinical metabolomics studies typically end with a biomarker question:
*can a small panel of metabolites separate cases from controls?*

Two functions:

- :func:`roc_feature` — per-feature AUC with optional bootstrap CI. Use
  for univariate screening; polarity-invariant (takes ``max(auc,
  1-auc)``) so you don't have to know *a priori* whether each
  metabolite should be up or down in cases.
- :func:`biomarker_panel` — multivariate nested cross-validation of a
  classifier on a candidate metabolite list (or top-N by univariate
  AUC pre-screen). Nested CV is the MetaboAnalyst-/FDA-recommended
  way to avoid over-optimistic AUC estimates. Supports RF / logistic
  regression / SVM plus an optional permutation null.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function


@register_function(
    aliases=[
        'roc_feature',
        'auc_feature',
        'per_feature_roc',
        '单变量ROC',
    ],
    category='metabolomics',
    description='Per-feature ROC AUC for a two-group comparison. Polarity-invariant (reports max(auc, 1-auc)). Optional bootstrap 95% CI.',
    examples=[
        "ov.metabol.roc_feature(adata, group_col='group')",
        "ov.metabol.roc_feature(adata, group_col='group', ci=True, n_bootstrap=1000)",
    ],
    related=[
        'metabol.biomarker_panel',
        'metabol.differential',
    ],
)
def roc_feature(
    adata: AnnData,
    *,
    group_col: str,
    pos_group: Optional[str] = None,
    neg_group: Optional[str] = None,
    layer: Optional[str] = None,
    ci: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-feature AUC for a binary class.

    Parameters
    ----------
    group_col
        Column in ``adata.obs`` with the class labels.
    pos_group, neg_group
        Label strings. If ``None`` the first two unique values in
        ``group_col`` are used (``neg_group`` first, ``pos_group``
        second — i.e. alphabetical/appearance order).
    layer
        AnnData layer name (default ``None`` → ``adata.X``).
    ci
        If True compute bootstrap 95% CI per feature. Costs
        ``n_bootstrap × n_features`` AUC evaluations; default ``False``.
    n_bootstrap
        Number of bootstrap resamples per feature. Default 1000.
    seed
        Bootstrap RNG seed.

    Returns
    -------
    pd.DataFrame
        Indexed by feature, sorted by AUC descending, with columns
        ``auc`` (and ``ci_low``, ``ci_high`` if ``ci=True``).
    """
    from sklearn.metrics import roc_auc_score

    groups = adata.obs[group_col].astype(str).to_numpy()
    if pos_group is None or neg_group is None:
        unique = list(pd.unique(adata.obs[group_col]))
        if len(unique) < 2:
            raise ValueError(f"{group_col!r} has fewer than 2 levels")
        neg_group = neg_group or str(unique[0])
        pos_group = pos_group or str(unique[1])

    mask = (groups == pos_group) | (groups == neg_group)
    if mask.sum() == 0:
        raise ValueError(
            f"no samples match pos/neg groups {pos_group!r}/{neg_group!r}"
        )
    y = (groups[mask] == pos_group).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("both classes must be present after masking")

    Xraw = adata.X if layer is None else adata.layers[layer]
    X = np.asarray(Xraw, dtype=np.float64)[mask]
    n, p = X.shape

    aucs = np.full(p, 0.5, dtype=np.float64)
    for j in range(p):
        xj = X[:, j]
        if not np.isfinite(xj).any() or len(np.unique(xj[np.isfinite(xj)])) < 2:
            continue
        try:
            a = float(roc_auc_score(y, np.nan_to_num(xj, nan=np.nanmedian(xj))))
        except Exception:
            continue
        aucs[j] = max(a, 1.0 - a)

    out = pd.DataFrame({"auc": aucs}, index=adata.var_names.copy())

    if ci:
        rng = np.random.default_rng(seed)
        lows = np.full(p, np.nan)
        highs = np.full(p, np.nan)
        for j in range(p):
            xj = X[:, j]
            if not np.isfinite(xj).any():
                continue
            xj_filled = np.nan_to_num(xj, nan=np.nanmedian(xj))
            bs = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, n)
                if len(np.unique(y[idx])) < 2:
                    continue
                try:
                    a = roc_auc_score(y[idx], xj_filled[idx])
                    bs.append(max(a, 1.0 - a))
                except Exception:
                    continue
            if bs:
                lows[j] = float(np.quantile(bs, 0.025))
                highs[j] = float(np.quantile(bs, 0.975))
        out["ci_low"] = lows
        out["ci_high"] = highs

    out = out.sort_values("auc", ascending=False)
    out.attrs["pos_group"] = pos_group
    out.attrs["neg_group"] = neg_group
    return out


@dataclass
class BiomarkerPanelResult:
    """Output of :func:`biomarker_panel` — nested-CV panel evaluation.

    Attributes
    ----------
    features : list[str]
        Metabolite names in the final panel.
    classifier : str
        ``"rf"``, ``"lr"``, or ``"svm"``.
    cv_outer, cv_inner : int
        Fold counts used in nested CV.
    outer_aucs : np.ndarray, (cv_outer,)
        Held-out AUC per outer fold.
    outer_predictions : np.ndarray, (n_samples,)
        Out-of-fold predicted probability / decision score per sample.
    outer_labels : np.ndarray, (n_samples,)
        Ground-truth labels (1 = pos_group, 0 = neg_group).
    feature_importance : pd.Series
        Mean importance across outer folds (RF: Gini; LR/SVM: |coef|).
    mean_auc, std_auc : float
        Aggregates of ``outer_aucs``.
    permutation_pvalue : float | None
        If ``n_permutations > 0``: empirical p from permuted labels.
    best_params_per_fold : list[dict]
        Hyperparameters selected by inner-CV for each outer fold.
    pos_group, neg_group : str
        Class labels used.
    """

    features: list[str]
    classifier: str
    cv_outer: int
    cv_inner: int
    outer_aucs: np.ndarray
    outer_predictions: np.ndarray
    outer_labels: np.ndarray
    feature_importance: pd.Series
    mean_auc: float
    std_auc: float
    permutation_pvalue: Optional[float]
    best_params_per_fold: list = field(default_factory=list)
    pos_group: str = ""
    neg_group: str = ""

    def to_frame(self) -> pd.DataFrame:
        """Per-fold AUC as a DataFrame (easy to paste into a report)."""
        return pd.DataFrame({"fold": np.arange(1, self.cv_outer + 1),
                             "auc": self.outer_aucs})


@register_function(
    aliases=[
        'biomarker_panel',
        'nested_cv',
        'roc_panel',
        '多变量生物标志物',
    ],
    category='metabolomics',
    description='Nested cross-validation of a multi-metabolite biomarker panel. RF/LR/SVM with inner-CV hyperparameter search + outer-CV unbiased AUC estimate, optional permutation null.',
    examples=[
        "ov.metabol.biomarker_panel(adata, group_col='group', features=10)",
        "ov.metabol.biomarker_panel(adata, group_col='group', features=['glucose','lactate'], classifier='lr', n_permutations=500)",
    ],
    related=[
        'metabol.roc_feature',
        'metabol.plsda',
        'metabol.opls_da',
    ],
)
def biomarker_panel(
    adata: AnnData,
    *,
    group_col: str,
    features: Union[list, int] = 10,
    classifier: str = "rf",
    cv_outer: int = 5,
    cv_inner: int = 3,
    pos_group: Optional[str] = None,
    neg_group: Optional[str] = None,
    n_permutations: int = 0,
    layer: Optional[str] = None,
    seed: int = 0,
) -> BiomarkerPanelResult:
    """Nested-CV evaluation of a multi-metabolite biomarker panel.

    Parameters
    ----------
    group_col
        Column in ``adata.obs`` with the class labels.
    features
        Either a list of metabolite names to use as-is, or an integer
        ``N`` to pre-screen the top-``N`` metabolites by univariate AUC
        **on the full dataset** (note: this leaks test-fold information
        — see "Caveats" below). Default 10.
    classifier
        ``"rf"`` (RandomForest, default), ``"lr"`` (L2-logistic
        regression), or ``"svm"`` (RBF SVM).
    cv_outer, cv_inner
        Stratified K-fold counts. Default 5-outer × 3-inner.
    pos_group, neg_group
        Class labels to contrast. ``None`` → first two unique values.
    n_permutations
        If > 0, compute a permutation null by shuffling labels and
        re-running nested CV that many times. Reported as
        ``permutation_pvalue``. Expensive (each permutation costs
        ``cv_outer × cv_inner`` fits).
    layer
        AnnData layer name (default ``None`` → ``adata.X``).
    seed
        RNG / fold-assignment seed.

    Returns
    -------
    BiomarkerPanelResult

    Caveats
    -------
    When ``features`` is an integer the pre-screen uses the full
    dataset, which overestimates AUC on the same folds. For
    publication-grade estimates pass an explicit feature list chosen
    from an independent screening cohort, or pre-screen inside a
    wrapper that repeats the full pipeline per fold.
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score

    groups = adata.obs[group_col].astype(str).to_numpy()
    if pos_group is None or neg_group is None:
        unique = list(pd.unique(adata.obs[group_col]))
        if len(unique) < 2:
            raise ValueError(f"{group_col!r} has fewer than 2 levels")
        neg_group = neg_group or str(unique[0])
        pos_group = pos_group or str(unique[1])

    mask = (groups == pos_group) | (groups == neg_group)
    y = (groups[mask] == pos_group).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("both classes must be present after masking")

    Xraw = adata.X if layer is None else adata.layers[layer]
    X_all = np.asarray(Xraw, dtype=np.float64)[mask]
    # Fill NaNs with per-column median so sklearn classifiers don't barf
    col_med = np.nanmedian(X_all, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    idx_nan = np.isnan(X_all)
    if idx_nan.any():
        X_all = np.where(idx_nan, np.broadcast_to(col_med, X_all.shape), X_all)

    if isinstance(features, int):
        aucs = np.full(X_all.shape[1], 0.5)
        for j in range(X_all.shape[1]):
            try:
                a = float(roc_auc_score(y, X_all[:, j]))
                aucs[j] = max(a, 1.0 - a)
            except Exception:
                pass
        order = np.argsort(aucs)[::-1][:features]
        feat_names = [str(adata.var_names[i]) for i in order]
        X = X_all[:, order]
    else:
        var_names = list(adata.var_names)
        want = list(features)
        miss = [f for f in want if f not in var_names]
        if miss:
            raise KeyError(f"features not in adata.var_names: {miss[:5]}...")
        idx = [var_names.index(f) for f in want]
        feat_names = want
        X = X_all[:, idx]

    # Variance guard — remove constant features (would break StandardScaler)
    nonconst = X.std(axis=0) > 1e-12
    if not nonconst.all():
        X = X[:, nonconst]
        feat_names = [f for f, keep in zip(feat_names, nonconst) if keep]
    if X.shape[1] == 0:
        raise ValueError("all selected features are constant after NaN-fill")

    def make_pipeline(cls_name: str):
        if cls_name == "rf":
            est = RandomForestClassifier(random_state=seed, n_estimators=200)
            grid = {"clf__max_depth": [3, 5, None]}
        elif cls_name == "lr":
            est = LogisticRegression(penalty="l2", solver="lbfgs",
                                     max_iter=500, random_state=seed)
            grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
        elif cls_name == "svm":
            est = SVC(kernel="rbf", probability=True, random_state=seed)
            grid = {"clf__C": [0.1, 1.0, 10.0]}
        else:
            raise ValueError(f"unknown classifier={cls_name!r}")
        pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
        return pipe, grid

    outer = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=seed)
    fold_aucs = np.zeros(cv_outer)
    preds = np.zeros(len(y))
    importance_sum = np.zeros(X.shape[1])
    best_params: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        pipe, grid = make_pipeline(classifier)
        inner = StratifiedKFold(n_splits=cv_inner, shuffle=True,
                                random_state=seed + fold)
        search = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc",
                              n_jobs=1)
        search.fit(X[train_idx], y[train_idx])
        best = search.best_estimator_
        best_params.append(dict(search.best_params_))
        clf = best.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            scores = best.predict_proba(X[test_idx])[:, 1]
        else:
            scores = best.decision_function(X[test_idx])
        preds[test_idx] = scores
        fold_aucs[fold] = float(roc_auc_score(y[test_idx], scores))

        if hasattr(clf, "feature_importances_"):
            importance_sum += clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importance_sum += np.abs(np.asarray(clf.coef_).ravel())

    feature_importance = pd.Series(importance_sum / cv_outer, index=feat_names)
    feature_importance = feature_importance.sort_values(ascending=False)
    feature_importance.name = "importance"

    perm_p: Optional[float] = None
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        observed = float(np.mean(fold_aucs))
        perm_scores = []
        for _ in range(n_permutations):
            y_perm = rng.permutation(y)
            aucs = []
            perm_outer = StratifiedKFold(n_splits=cv_outer, shuffle=True,
                                         random_state=int(rng.integers(0, 2**31 - 1)))
            for tr, te in perm_outer.split(X, y_perm):
                pipe, grid = make_pipeline(classifier)
                perm_inner = StratifiedKFold(n_splits=cv_inner, shuffle=True,
                                             random_state=seed)
                try:
                    s = GridSearchCV(pipe, grid, cv=perm_inner,
                                     scoring="roc_auc", n_jobs=1)
                    s.fit(X[tr], y_perm[tr])
                    est = s.best_estimator_
                    clf = est.named_steps["clf"]
                    scores = (est.predict_proba(X[te])[:, 1]
                              if hasattr(clf, "predict_proba")
                              else est.decision_function(X[te]))
                    aucs.append(float(roc_auc_score(y_perm[te], scores)))
                except Exception:
                    continue
            if aucs:
                perm_scores.append(float(np.mean(aucs)))
        if perm_scores:
            perm_p = float((np.sum(np.asarray(perm_scores) >= observed) + 1)
                           / (len(perm_scores) + 1))

    # Build a full-length out-of-fold prediction vector aligned to the
    # **filtered** samples (matches outer_labels).
    return BiomarkerPanelResult(
        features=feat_names,
        classifier=classifier,
        cv_outer=cv_outer,
        cv_inner=cv_inner,
        outer_aucs=fold_aucs,
        outer_predictions=preds,
        outer_labels=y,
        feature_importance=feature_importance,
        mean_auc=float(np.mean(fold_aucs)),
        std_auc=float(np.std(fold_aucs)),
        permutation_pvalue=perm_p,
        best_params_per_fold=best_params,
        pos_group=pos_group,
        neg_group=neg_group,
    )
