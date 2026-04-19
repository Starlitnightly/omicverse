r"""Multi-factor designs for metabolomics — ASCA + linear mixed models.

Most metabolomics studies are not two-group comparisons. Typical designs:

- ``treatment × time`` — paired pre/post or dose-response
- ``patient × visit`` — repeated measures
- ``batch × treatment`` — nuisance + effect of interest

Two complementary tools:

- :func:`asca` — ANOVA-Simultaneous Component Analysis (Smilde 2005).
  Decomposes the data matrix into per-factor effect matrices (plus
  pairwise interactions and a residual), runs PCA on each, and reports
  variance-explained plus permutation-based significance per effect.
  This is MetaboAnalyst's multifactor module with matching output.
- :func:`mixed_model` — per-feature linear mixed model via
  ``statsmodels.MixedLM``. The right choice when samples are
  non-independent (patient ID → random intercept), as it models the
  within-subject correlation explicitly.

The two methods answer different questions: ``asca`` asks *which global
pattern is attributable to each factor?* ``mixed_model`` asks *for
each metabolite, what is the effect size and p-value while accounting
for the random structure?*
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function
from ._utils import bh_fdr


# ---------------------------------------------------------------------------
# ASCA
# ---------------------------------------------------------------------------
@dataclass
class ASCAEffect:
    """Per-effect output of :func:`asca` (main effect or pairwise interaction).

    Attributes
    ----------
    name : str
        e.g. ``"treatment"`` or ``"treatment:time"``.
    effect_matrix : np.ndarray, (n_samples, n_features)
        The decomposed effect contribution — rows sum (with other effects
        and the residual) to the centered ``X``.
    scores : np.ndarray, (n_samples, n_components)
        PCA scores of the effect matrix. Suitable for a 2-D plot of
        samples coloured by the factor.
    loadings : np.ndarray, (n_features, n_components)
        PCA loadings — which metabolites drive the effect.
    variance_explained : float
        Fraction of total SS (centered ``X``) captured by this effect.
    ss : float
        Sum of squares of ``effect_matrix``.
    df : int
        Degrees of freedom — ``k-1`` for a main effect with ``k`` levels,
        ``(k_a-1)(k_b-1)`` for an ``a×b`` interaction.
    p_value : float | None
        Permutation p-value, or ``None`` when ``n_permutations=0``.
    """

    name: str
    effect_matrix: np.ndarray
    scores: np.ndarray
    loadings: np.ndarray
    singular_values: np.ndarray
    variance_explained: float
    component_variance_ratio: np.ndarray
    ss: float
    df: int
    p_value: Optional[float] = None


@dataclass
class ASCAResult:
    """Full output of :func:`asca`.

    Access per-effect data via ``result.effects['treatment']``. Call
    :meth:`summary` for a one-row-per-effect DataFrame.
    """

    effects: dict[str, ASCAEffect]
    residual_ss: float
    total_ss: float
    sample_names: pd.Index
    var_names: pd.Index
    factors: list[str]
    n_permutations: int

    def summary(self) -> pd.DataFrame:
        """One row per effect + residual, with SS / df / variance / p."""
        rows = []
        for name, e in self.effects.items():
            rows.append({
                "effect": name,
                "ss": e.ss,
                "df": e.df,
                "variance_explained": e.variance_explained,
                "p_value": e.p_value,
            })
        rows.append({
            "effect": "residual",
            "ss": self.residual_ss,
            "df": np.nan,
            "variance_explained": (self.residual_ss / self.total_ss
                                   if self.total_ss > 0 else np.nan),
            "p_value": np.nan,
        })
        return pd.DataFrame(rows)

    def scores_frame(self, effect: str) -> pd.DataFrame:
        """Scores of one effect as a DataFrame indexed by sample."""
        e = self.effects[effect]
        cols = [f"PC{i+1}" for i in range(e.scores.shape[1])]
        return pd.DataFrame(e.scores, index=self.sample_names, columns=cols)

    def loadings_frame(self, effect: str) -> pd.DataFrame:
        """Loadings of one effect as a DataFrame indexed by feature."""
        e = self.effects[effect]
        cols = [f"PC{i+1}" for i in range(e.loadings.shape[1])]
        return pd.DataFrame(e.loadings, index=self.var_names, columns=cols)


@register_function(
    aliases=[
        'asca',
        'ASCA',
        'multifactor',
        '多因素代谢组',
    ],
    category='metabolomics',
    description='ANOVA-Simultaneous Component Analysis (Smilde 2005) — decompose metabolomics X into per-factor + interaction effect matrices, run PCA on each, report variance-explained and permutation p.',
    examples=[
        "ov.metabol.asca(adata, factors=['treatment', 'time'])",
        "ov.metabol.asca(adata, factors=['treatment', 'time'], n_permutations=500)",
    ],
    related=[
        'metabol.mixed_model',
        'metabol.differential',
        'metabol.plsda',
    ],
)
def asca(
    adata: AnnData,
    *,
    factors: Sequence[str],
    include_interactions: bool = True,
    n_components: int = 2,
    n_permutations: int = 0,
    layer: Optional[str] = None,
    center: bool = True,
    seed: int = 0,
) -> ASCAResult:
    """ASCA — ANOVA-Simultaneous Component Analysis (Smilde 2005).

    Decomposes the mean-centered data matrix into per-factor effect
    matrices plus (optionally) pairwise interactions and a residual,
    runs PCA on each effect, and reports variance explained and
    (optionally) a permutation p-value.

    Parameters
    ----------
    factors
        1–3 column names in ``adata.obs``. Each is treated categorically.
    include_interactions
        Whether to add pairwise ``A:B`` interaction terms. Three-way
        interactions are not computed here — in practice they require
        far more replicates than a typical metabolomics design
        supports.
    n_components
        Number of PCs to retain per effect. Default 2 — enough for the
        standard ASCA scatter plot.
    n_permutations
        If > 0, shuffle each factor's labels that many times and compute
        the null distribution of effect SS. Reported as ``p_value``.
        Default 0 (fast; no p-value).
    layer
        AnnData layer name (default ``None`` → ``adata.X``).
    center
        If True (default) subtract the grand mean before decomposition.
        ASCA as described in Smilde 2005 assumes centered data.
    seed
        RNG seed for permutation.

    Returns
    -------
    ASCAResult
        Access effects via ``result.effects[name]`` or call
        :meth:`ASCAResult.summary`. Names are factor names
        (e.g. ``"treatment"``) or ``"A:B"`` for interactions.
    """
    if not factors:
        raise ValueError("need at least one factor")
    for f in factors:
        if f not in adata.obs.columns:
            raise KeyError(f"adata.obs has no column {f!r}")

    Xraw = adata.X if layer is None else adata.layers[layer]
    X = np.asarray(Xraw, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    total_ss = float((X ** 2).sum())

    factor_labels = {f: adata.obs[f].astype(str).to_numpy() for f in factors}

    main_terms = list(factors)
    interaction_terms: list[tuple[str, str]] = []
    if include_interactions and len(factors) >= 2:
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                interaction_terms.append((factors[i], factors[j]))

    effect_matrices: dict[str, np.ndarray] = {}
    residual = X.copy()

    for f in main_terms:
        E = _main_effect_matrix(X, factor_labels[f])
        effect_matrices[f] = E
        residual = residual - E

    for (f, g) in interaction_terms:
        name = f"{f}:{g}"
        cell = _cell_mean_matrix(X, [factor_labels[f], factor_labels[g]])
        E = cell - effect_matrices[f] - effect_matrices[g]
        effect_matrices[name] = E
        residual = residual - E

    residual_ss = float((residual ** 2).sum())

    perm_ss: dict[str, list[float]] = {name: [] for name in effect_matrices}
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_permutations):
            shuffled = {f: rng.permutation(factor_labels[f]) for f in factors}
            tmp_main: dict[str, np.ndarray] = {}
            for f in main_terms:
                E = _main_effect_matrix(X, shuffled[f])
                tmp_main[f] = E
                perm_ss[f].append(float((E ** 2).sum()))
            for (f, g) in interaction_terms:
                cell = _cell_mean_matrix(X, [shuffled[f], shuffled[g]])
                E = cell - tmp_main[f] - tmp_main[g]
                perm_ss[f"{f}:{g}"].append(float((E ** 2).sum()))

    effects: dict[str, ASCAEffect] = {}
    for name, E in effect_matrices.items():
        ss = float((E ** 2).sum())
        k = max(1, min(n_components, min(E.shape)))
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        k = min(k, len(S))
        scores = U[:, :k] * S[:k]
        loadings = Vt[:k, :].T
        total_s2 = float(np.sum(S ** 2))
        comp_var = ((S[:k] ** 2) / total_s2) if total_s2 > 0 else np.zeros(k)

        pval: Optional[float] = None
        if n_permutations > 0:
            null = np.asarray(perm_ss[name], dtype=np.float64)
            pval = float((np.sum(null >= ss) + 1) / (n_permutations + 1))

        if ":" in name:
            f, g = name.split(":")
            df = ((len(np.unique(factor_labels[f])) - 1)
                  * (len(np.unique(factor_labels[g])) - 1))
        else:
            df = len(np.unique(factor_labels[name])) - 1

        effects[name] = ASCAEffect(
            name=name,
            effect_matrix=E,
            scores=scores,
            loadings=loadings,
            singular_values=S[:k].astype(np.float64),
            variance_explained=(ss / total_ss if total_ss > 0 else 0.0),
            component_variance_ratio=comp_var.astype(np.float64),
            ss=ss,
            df=int(df),
            p_value=pval,
        )

    return ASCAResult(
        effects=effects,
        residual_ss=residual_ss,
        total_ss=total_ss,
        sample_names=adata.obs_names.copy(),
        var_names=adata.var_names.copy(),
        factors=list(factors),
        n_permutations=n_permutations,
    )


def _main_effect_matrix(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample row = mean of all samples sharing the same level."""
    out = np.zeros_like(X)
    for lvl in np.unique(labels):
        mask = labels == lvl
        out[mask] = X[mask].mean(axis=0, keepdims=True)
    return out


def _cell_mean_matrix(X: np.ndarray, label_lists: list[np.ndarray]) -> np.ndarray:
    """Per-sample row = mean across all samples sharing the same
    combination of labels across the supplied factor arrays."""
    n = X.shape[0]
    keys = np.array([
        " | ".join(str(ll[i]) for ll in label_lists)
        for i in range(n)
    ])
    out = np.zeros_like(X)
    for key in np.unique(keys):
        mask = keys == key
        out[mask] = X[mask].mean(axis=0, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# MixedLM
# ---------------------------------------------------------------------------
@register_function(
    aliases=[
        'mixed_model',
        'mixedlm',
        'lmm',
        '线性混合模型',
    ],
    category='metabolomics',
    description='Per-feature linear mixed model via statsmodels.MixedLM. Right tool for repeated-measures / longitudinal designs where samples share a random structure (patient ID, site).',
    examples=[
        "ov.metabol.mixed_model(adata, formula='treatment + time', groups='patient')",
        "ov.metabol.mixed_model(adata, formula='treatment * time', groups='patient', term='treatment[T.drug]')",
    ],
    related=[
        'metabol.asca',
        'metabol.differential',
    ],
)
def mixed_model(
    adata: AnnData,
    *,
    formula: str,
    groups: str,
    re_formula: Optional[str] = "1",
    term: Optional[str] = None,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """Per-feature ``statsmodels.MixedLM`` fit.

    Fits ``y ~ <formula>`` for each metabolite with ``groups=<groups>``
    defining the random-effect grouping variable (e.g. patient ID).

    Parameters
    ----------
    formula
        Patsy-style formula **without** the ``y ~`` prefix. Categorical
        effects, interactions (``a * b``), and numerics all work.
    groups
        Name of the ``adata.obs`` column carrying the random-effect
        grouping labels. Usually patient / subject ID.
    re_formula
        Random-effect formula. ``"1"`` (default) = random intercept.
        Use ``"1 + time"`` for random slopes.
    term
        If given, return a short-format table with one row per feature
        for this specific fixed-effect term (matches the
        :func:`metabol.differential` schema: ``stat / pvalue / padj``).
        If ``None``, return a long-format table with one row per
        ``(feature, term)`` pair, excluding the intercept and variance
        components.
    layer
        AnnData layer name (default ``None`` → ``adata.X``).

    Returns
    -------
    pd.DataFrame
        Long format (default): columns ``feature, term, coef, se, stat,
        pvalue, padj``. Short format (``term=...``): indexed by feature
        with ``coef, se, stat, pvalue, padj``.
    """
    import statsmodels.formula.api as smf

    if groups not in adata.obs.columns:
        raise KeyError(f"adata.obs has no column {groups!r}")
    Xraw = adata.X if layer is None else adata.layers[layer]
    X = np.asarray(Xraw, dtype=np.float64)
    obs = adata.obs.copy()
    n, p = X.shape
    var_names = list(adata.var_names)

    rows: list[dict] = []
    for j in range(p):
        df_j = obs.copy()
        df_j["_y_"] = X[:, j]
        model_formula = f"_y_ ~ {formula}"
        try:
            model = smf.mixedlm(model_formula, df_j, groups=df_j[groups],
                                re_formula=re_formula)
            res = model.fit(method="lbfgs", disp=False)
            params = res.params
            ses = res.bse
            tvals = res.tvalues
            pvals = res.pvalues
        except Exception:
            fallback_term = term if term is not None else "fit_failed"
            rows.append({"feature": var_names[j], "term": fallback_term,
                         "coef": np.nan, "se": np.nan,
                         "stat": np.nan, "pvalue": np.nan})
            continue

        if term is not None:
            if term not in params.index:
                raise KeyError(
                    f"term {term!r} not found in MixedLM result; "
                    f"available: {list(params.index)}"
                )
            rows.append({"feature": var_names[j], "term": term,
                         "coef": float(params[term]),
                         "se": float(ses[term]),
                         "stat": float(tvals[term]),
                         "pvalue": float(pvals[term])})
        else:
            for name in params.index:
                # Skip intercept and the random-effect variance component
                if name == "Intercept" or name.startswith("Group "):
                    continue
                rows.append({"feature": var_names[j], "term": name,
                             "coef": float(params[name]),
                             "se": float(ses[name]),
                             "stat": float(tvals[name]),
                             "pvalue": float(pvals[name])})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # BH FDR per term
    out["padj"] = np.nan
    for t in out["term"].unique():
        mask = (out["term"] == t).to_numpy()
        pvals = out.loc[mask, "pvalue"].to_numpy()
        out.loc[mask, "padj"] = bh_fdr(pvals)

    if term is not None:
        # Short format: index by feature
        out = out.set_index("feature")
        out = out[["coef", "se", "stat", "pvalue", "padj"]]
        out.attrs["term"] = term
        out.attrs["formula"] = formula
        out.attrs["groups"] = groups
    return out
