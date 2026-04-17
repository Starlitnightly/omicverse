from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from pandas.api.types import infer_dtype, is_numeric_dtype, is_object_dtype
from scipy.sparse import issparse
from tqdm.auto import tqdm

from .._registry import register_function
from .._settings import Colors, EMOJI


def _require_pygam():
    try:
        from pygam import GammaGAM, LinearGAM, LogisticGAM, PoissonGAM, s
    except ImportError as e:
        raise ImportError(
            "dynamic_features requires the optional `pygam` package. "
            "Install it with `pip install pygam`."
        ) from e

    return {
        ("normal", "identity"): LinearGAM,
        ("gamma", "log"): GammaGAM,
        ("poisson", "log"): PoissonGAM,
        ("binomial", "logit"): LogisticGAM,
        "s": s,
    }


def _as_dataset_map(data: AnnData | Mapping[str, AnnData]) -> dict[str, AnnData]:
    if isinstance(data, AnnData):
        return {"adata": data}
    if isinstance(data, Mapping):
        out = {}
        for key, value in data.items():
            if not isinstance(value, AnnData):
                raise TypeError(
                    "When `data` is a mapping, all values must be AnnData objects. "
                    f"Found `{type(value).__name__}` for key `{key}`."
                )
            out[str(key)] = value
        if not out:
            raise ValueError("`data` mapping is empty.")
        return out
    raise TypeError("`data` must be an AnnData object or a mapping of name -> AnnData.")


def _resolve_subset(subsets: Any, dataset_name: str):
    if subsets is None:
        return None
    if isinstance(subsets, Mapping):
        return subsets.get(dataset_name)
    return subsets


def _resolve_dataset_param(value: Any, dataset_name: str, source_dataset_name: str):
    if not isinstance(value, Mapping):
        return value
    if dataset_name in value:
        return value.get(dataset_name)
    return value.get(source_dataset_name)


def _iter_dataset_views(
    dataset_map: Mapping[str, AnnData],
    *,
    groupby: str | Mapping[str, str] | None = None,
    groups: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
):
    multi_dataset = len(dataset_map) > 1
    for dataset_name, adata in dataset_map.items():
        groupby_key = _resolve_subset(groupby, dataset_name)
        if groupby_key is None:
            yield {
                "dataset": dataset_name,
                "source_dataset": dataset_name,
                "groupby_key": None,
                "group": None,
                "adata": adata,
            }
            continue
        if groupby_key not in adata.obs.columns:
            raise KeyError(f"Group key `{groupby_key}` was not found in adata.obs for dataset `{dataset_name}`.")

        group_series = adata.obs[groupby_key]
        allowed_groups = _resolve_subset(groups, dataset_name)
        if allowed_groups is not None:
            allowed_groups = {str(group) for group in allowed_groups}
        present_groups = []
        for value in group_series:
            if pd.isna(value):
                continue
            label = str(value)
            if allowed_groups is not None and label not in allowed_groups:
                continue
            if label not in present_groups:
                present_groups.append(label)
        if not present_groups:
            raise ValueError(f"No groups remained for dataset `{dataset_name}` after applying `groups`.")

        for group_name in present_groups:
            mask = group_series.astype(str).to_numpy() == group_name
            if not np.any(mask):
                continue
            dataset_label = f"{dataset_name}:{group_name}" if multi_dataset else group_name
            yield {
                "dataset": dataset_label,
                "source_dataset": dataset_name,
                "groupby_key": groupby_key,
                "group": group_name,
                "adata": adata[mask],
            }


def _resolve_weights(weights: Any, adata: AnnData, dataset_name: str):
    if weights is None:
        return None
    if isinstance(weights, Mapping):
        weights = weights.get(dataset_name)
    if weights is None:
        return None
    if isinstance(weights, str):
        if weights not in adata.obs.columns:
            raise KeyError(f"Weight key `{weights}` was not found in adata.obs for dataset `{dataset_name}`.")
        return pd.to_numeric(adata.obs[weights], errors="coerce").to_numpy(dtype=float)
    arr = np.asarray(weights, dtype=float)
    if arr.shape[0] != adata.n_obs:
        raise ValueError(
            f"Weights for dataset `{dataset_name}` must have length `{adata.n_obs}`, found `{arr.shape[0]}`."
        )
    return arr


def _available_feature_names(adata: AnnData, use_raw: bool = False) -> set[str]:
    available = set(adata.raw.var_names if use_raw and adata.raw is not None else adata.var_names)
    numeric_obs = {col for col in adata.obs.columns if is_numeric_dtype(adata.obs[col])}
    return available | numeric_obs


def _resolve_gene_list(dataset_map: Mapping[str, AnnData], genes, use_raw: bool = False) -> list[str]:
    if genes is None:
        shared = None
        for adata in dataset_map.values():
            names = _available_feature_names(adata, use_raw=use_raw)
            shared = names if shared is None else shared & names
        if not shared:
            raise ValueError("No shared numeric features were found across datasets.")
        return sorted(shared)

    if isinstance(genes, str):
        genes = [genes]
    genes = [str(gene) for gene in genes]
    if not genes:
        raise ValueError("`genes` is empty.")
    return genes


def _extract_feature_vector(
    adata: AnnData,
    feature: str,
    *,
    layer: str | None = None,
    use_raw: bool = False,
) -> np.ndarray:
    if feature in adata.obs.columns:
        values = pd.to_numeric(adata.obs[feature], errors="coerce").to_numpy(dtype=float)
        return values

    if use_raw:
        if adata.raw is None:
            raise ValueError("`use_raw=True` was requested, but adata.raw is not available.")
        if feature not in adata.raw.var_names:
            raise KeyError(f"Feature `{feature}` not found in adata.raw.var_names.")
        values = adata.raw[:, [feature]].X
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer `{layer}` not found in adata.layers.")
        if feature not in adata.var_names:
            raise KeyError(f"Feature `{feature}` not found in adata.var_names.")
        values = adata[:, [feature]].layers[layer]
    else:
        if feature not in adata.var_names:
            raise KeyError(f"Feature `{feature}` not found in adata.var_names.")
        values = adata[:, [feature]].X

    if issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=float).reshape(-1)


def _pseudo_r2_from_stats(stats_dict: Mapping[str, Any]) -> float:
    pseudo_r2 = stats_dict.get("pseudo_r2")
    if isinstance(pseudo_r2, Mapping):
        for key in ("explained_deviance", "McFadden", "McFadden_adj", "pseudo_r2"):
            value = pseudo_r2.get(key)
            if value is not None and np.isfinite(value):
                return float(value)
    if pseudo_r2 is not None and np.isscalar(pseudo_r2) and np.isfinite(pseudo_r2):
        return float(pseudo_r2)
    return np.nan


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _summarize_trend(x_pred: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    peak_cutoff = np.nanquantile(y_pred, 0.99) if len(y_pred) > 1 else np.nanmax(y_pred)
    valley_cutoff = np.nanquantile(y_pred, 0.01) if len(y_pred) > 1 else np.nanmin(y_pred)
    peak_time = float(np.nanmedian(x_pred[y_pred >= peak_cutoff]))
    valley_time = float(np.nanmedian(x_pred[y_pred <= valley_cutoff]))
    return peak_time, valley_time


def _expressed_cells(y: np.ndarray) -> int:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return 0
    if np.nanmin(finite) >= 0:
        return int(np.sum(finite > 0))
    baseline = np.nanmin(finite)
    return int(np.sum(finite > baseline))


def _prepare_table_for_uns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a result table with h5ad-friendly string columns.

    AnnData/HDF5 cannot reliably serialize object-dtype columns that mix
    ``None`` with Python strings. Normalize purely string-like object columns to
    plain Python string objects before storing under ``adata.uns``.
    """
    out = df.copy()
    for column in out.columns:
        series = out[column]
        if not is_object_dtype(series.dtype):
            continue
        kind = infer_dtype(series, skipna=True)
        if kind in {"string", "unicode", "bytes", "empty"}:
            out[column] = series.map(lambda value: "" if pd.isna(value) else str(value))
    return out


@dataclass
class DynamicFeaturesResult:
    """Container returned by :func:`dynamic_features`.

    Attributes
    ----------
    stats
        Per-feature goodness-of-fit dataframe across groups (R², AIC,
        deviance, significance flags).
    fitted
        Long-form dataframe of predicted trend curves — one row per
        ``(dataset, group, feature, x_pred point)``. Ready to pass to
        :func:`omicverse.pl.dynamic_trends`.
    raw
        Per-cell raw values in the same long form as ``fitted`` when
        ``keep_raw=True`` was passed; ``None`` otherwise.
    models
        Mapping of ``(group, feature)`` to the fitted pyGAM model for
        users who want to inspect coefficients or re-predict.
    config
        Echo of the parameters the call ran with, for reproducibility.
    """

    stats: pd.DataFrame
    fitted: pd.DataFrame
    raw: pd.DataFrame | None = None
    models: dict[tuple[str, str], Any] | None = None
    config: dict[str, Any] | None = None

    def get_stats(self, genes=None, datasets=None, successful_only: bool = False) -> pd.DataFrame:
        df = self.stats.copy()
        if genes is not None:
            genes = [genes] if isinstance(genes, str) else list(genes)
            df = df[df["gene"].isin(genes)]
        if datasets is not None:
            datasets = [datasets] if isinstance(datasets, str) else list(datasets)
            df = df[df["dataset"].isin(datasets)]
        if successful_only and "success" in df.columns:
            df = df[df["success"]]
        return df

    def get_fitted(self, genes=None, datasets=None) -> pd.DataFrame:
        df = self.fitted.copy()
        if genes is not None:
            genes = [genes] if isinstance(genes, str) else list(genes)
            df = df[df["gene"].isin(genes)]
        if datasets is not None:
            datasets = [datasets] if isinstance(datasets, str) else list(datasets)
            df = df[df["dataset"].isin(datasets)]
        return df

    def get_raw(self, genes=None, datasets=None) -> pd.DataFrame | None:
        if self.raw is None:
            return None
        df = self.raw.copy()
        if genes is not None:
            genes = [genes] if isinstance(genes, str) else list(genes)
            df = df[df["gene"].isin(genes)]
        if datasets is not None:
            datasets = [datasets] if isinstance(datasets, str) else list(datasets)
            df = df[df["dataset"].isin(datasets)]
        return df

    def get_significant_features(
        self,
        *,
        datasets=None,
        min_expcells: int | None = None,
        r2_cutoff: float | None = None,
        dev_expl_cutoff: float | None = None,
        padj_cutoff: float | None = None,
        as_dict: bool = False,
    ):
        df = self.get_stats(datasets=datasets, successful_only=True)
        if min_expcells is not None and "exp_ncells" in df.columns:
            df = df[df["exp_ncells"] >= int(min_expcells)]
        if r2_cutoff is not None and "r2" in df.columns:
            df = df[df["r2"] >= float(r2_cutoff)]
        if dev_expl_cutoff is not None and "explained_deviance" in df.columns:
            df = df[df["explained_deviance"] >= float(dev_expl_cutoff)]
        if padj_cutoff is not None and "padj" in df.columns:
            df = df[df["padj"] <= float(padj_cutoff)]

        if as_dict:
            return {
                dataset: sub["gene"].dropna().astype(str).tolist()
                for dataset, sub in df.groupby("dataset", sort=False)
            }
        return df["gene"].dropna().astype(str).tolist()


@register_function(
    aliases=["dynamic_features", "动态特征", "动态基因", "gam_dynamic_features"],
    category="single",
    description="Fit GAM-based pseudotime trends for one or multiple AnnData objects and return reusable dynamic-feature statistics and fitted curves.",
    requires={"obs": ["pseudotime"], "var": ["genes/features"]},
    produces={"uns": ["dynamic feature tables", "dynamic fitted trends"]},
    auto_fix="none",
    examples=[
        "res = ov.single.dynamic_features(adata, genes=['Sox9', 'Neurog3'], pseudotime='palantir_pseudotime')",
        "res = ov.single.dynamic_features({'MHWA': rna_mhwa, 'Normal': rna_normal}, genes=['tbxta'], pseudotime='palantir_pseudotime')",
    ],
    related=["pl.dynamic_trends", "pl.dynamic_heatmap", "single.TrajInfer"],
)
def dynamic_features(
    data: AnnData | Mapping[str, AnnData],
    genes: Sequence[str] | str | None = None,
    pseudotime: str = "pseudotime",
    *,
    groupby: str | Mapping[str, str] | None = None,
    groups: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    subsets: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    weights: str | Sequence[float] | Mapping[str, Any] | None = None,
    distribution: str = "normal",
    link: str = "identity",
    n_splines: int = 8,
    spline_order: int = 3,
    grid_size: int = 200,
    confidence_level: float = 0.95,
    min_cells: int = 20,
    min_variance: float = 1e-8,
    store_raw: bool = False,
    key_added: str | None = "dynamic_features",
    verbose: bool = True,
) -> DynamicFeaturesResult:
    """Fit GAM-based pseudotime trends for one or more datasets or groups.

    Parameters
    ----------
    data
        A single :class:`~anndata.AnnData` object or a mapping of dataset names
        to AnnData objects.
    genes
        Feature names to model. When ``None``, genes are inferred from the
        available features in the supplied datasets.
    pseudotime
        Column in ``adata.obs`` containing pseudotime values.
    groupby
        Optional ``obs`` column used to split each dataset into separate trend
        groups, such as cell types or lineages. Each group is fitted as its own
        comparable series. The chosen key is recorded in the result tables as
        ``groupby_key``.
    groups
        Optional subset of group labels to keep when ``groupby`` is provided.
    layer
        Optional layer name used as the expression source.
    use_raw
        Whether to read expression values from ``adata.raw``.
    subsets
        Optional cell subsets to retain before fitting, either globally or per
        dataset.
    weights
        Optional sample weights, provided as an obs key, array-like values, or
        per-dataset mapping. For grouped or multi-dataset inputs, mappings may
        be keyed by the final plotted dataset label or by the source dataset
        name.
    distribution
        GAM response distribution passed to :mod:`pygam`.
    link
        Link function paired with ``distribution``.
    n_splines
        Number of spline basis functions.
    spline_order
        Order of the spline basis.
    grid_size
        Number of pseudotime grid points used to evaluate fitted curves.
    confidence_level
        Confidence level used to compute fitted intervals.
    min_cells
        Minimum number of valid cells required to fit a model.
    min_variance
        Minimum variance required for a feature to be modeled.
    store_raw
        Whether to include per-cell raw observations in the returned result.
        This must be enabled if downstream plotting should overlay individual
        points, for example via :func:`ov.pl.dynamic_trends` with
        ``add_point=True``.
    key_added
        Optional key used to store outputs in ``adata.uns`` for single-dataset
        inputs.
    verbose
        Whether to print progress information during fitting.

    Returns
    -------
    DynamicFeaturesResult
        Container holding fitted trends, summary statistics, and optional raw
        observations for downstream visualization and filtering. The returned
        tables use:

        ``dataset``
            Final plotted series label used for comparison.
        ``source_dataset``
            Original dataset key before any group splitting. This column is
            included only when the input contains multiple datasets.
        ``groupby_key``
            The ``obs`` column used for grouping, if any.
        ``group``
            The concrete group label (for example a cell type or lineage).
    """
    dataset_map = _as_dataset_map(data)
    include_source_dataset = len(dataset_map) > 1
    gene_list = _resolve_gene_list(dataset_map, genes, use_raw=use_raw)
    pygam_parts = _require_pygam()
    gam_class = pygam_parts.get((str(distribution).lower(), str(link).lower()))
    if gam_class is None:
        valid = ", ".join(sorted(k for k in pygam_parts if isinstance(k, tuple)))
        raise ValueError(f"Unsupported (distribution, link)=({distribution!r}, {link!r}). Supported: {valid}.")
    spline_term = pygam_parts["s"]

    stats_records: list[dict[str, Any]] = []
    fitted_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    model_store: dict[tuple[str, str], Any] = {}

    dataset_views = list(_iter_dataset_views(dataset_map, groupby=groupby, groups=groups))
    total_jobs = len(dataset_views) * len(gene_list)
    if verbose:
        print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Dynamic feature analysis:{Colors.ENDC}")
        print(f"   {Colors.CYAN}Views: {Colors.BOLD}{len(dataset_views)}{Colors.ENDC}{Colors.CYAN} | Features: {Colors.BOLD}{len(gene_list)}{Colors.ENDC}")
        print(f"   {Colors.CYAN}Pseudotime: {Colors.BOLD}{pseudotime}{Colors.ENDC}")
        if groupby is not None:
            print(f"   {Colors.CYAN}Grouping: {Colors.BOLD}{groupby}{Colors.ENDC}")
        if layer is not None:
            print(f"   {Colors.CYAN}Layer: {Colors.BOLD}{layer}{Colors.ENDC}")
        elif use_raw:
            print(f"   {Colors.CYAN}Expression source: {Colors.BOLD}adata.raw{Colors.ENDC}")
        print(f"   {Colors.CYAN}GAM: {Colors.BOLD}{distribution}-{link}{Colors.ENDC}{Colors.CYAN} | splines={Colors.BOLD}{n_splines}{Colors.ENDC}")
    progress = tqdm(
        total=total_jobs,
        desc="Fitting dynamic features",
        disable=(not verbose or total_jobs <= 1),
        leave=False,
    )

    for dataset_view in dataset_views:
        dataset_name = dataset_view["dataset"]
        source_dataset_name = dataset_view["source_dataset"]
        groupby_key = dataset_view["groupby_key"]
        group_name = dataset_view["group"]
        adata = dataset_view["adata"]
        subset = _resolve_dataset_param(subsets, dataset_name, source_dataset_name)
        adata_use = adata[subset].copy() if subset is not None else adata
        if pseudotime not in adata_use.obs.columns:
            raise KeyError(f"Pseudotime key `{pseudotime}` was not found in adata.obs for dataset `{dataset_name}`.")

        time = pd.to_numeric(adata_use.obs[pseudotime], errors="coerce").to_numpy(dtype=float)
        weight_spec = _resolve_dataset_param(weights, dataset_name, source_dataset_name)
        weights_arr = _resolve_weights(weight_spec, adata_use, dataset_name)
        keep = np.isfinite(time)
        if np.sum(keep) < min_cells:
            raise ValueError(
                f"Dataset `{dataset_name}` has fewer than `{min_cells}` finite pseudotime values after filtering."
            )
        if not np.all(keep):
            adata_use = adata_use[keep].copy()
            time = time[keep]
            if weights_arr is not None:
                weights_arr = weights_arr[keep]
        order = np.argsort(time, kind="stable")
        time = time[order]
        available_features = _available_feature_names(adata_use, use_raw=use_raw)

        for gene in gene_list:
            if gene not in available_features:
                stats_records.append(
                    {
                        "dataset": dataset_name,
                        "source_dataset": source_dataset_name,
                        "groupby_key": groupby_key,
                        "group": group_name,
                        "gene": gene,
                        "success": False,
                        "error": f"Feature `{gene}` not found.",
                    }
                )
                progress.update(1)
                continue

            y = _extract_feature_vector(adata_use, gene, layer=layer, use_raw=use_raw)[order]
            w = weights_arr[order] if weights_arr is not None else None

            finite = np.isfinite(time) & np.isfinite(y)
            if w is not None:
                finite &= np.isfinite(w)
            x_fit = time[finite]
            y_fit = y[finite]
            w_fit = w[finite] if w is not None else None

            if x_fit.shape[0] < min_cells:
                stats_records.append(
                    {
                        "dataset": dataset_name,
                        "source_dataset": source_dataset_name,
                        "groupby_key": groupby_key,
                        "group": group_name,
                        "gene": gene,
                        "success": False,
                        "error": f"Only `{x_fit.shape[0]}` valid cells remained after filtering.",
                    }
                )
                progress.update(1)
                continue
            if np.nanvar(y_fit) <= float(min_variance):
                stats_records.append(
                    {
                        "dataset": dataset_name,
                        "source_dataset": source_dataset_name,
                        "groupby_key": groupby_key,
                        "group": group_name,
                        "gene": gene,
                        "success": False,
                        "error": "Feature variance is too small for GAM fitting.",
                    }
                )
                progress.update(1)
                continue

            model = gam_class(spline_term(0, n_splines=max(int(n_splines), int(spline_order) + 1), spline_order=int(spline_order)))
            try:
                model.fit(x_fit[:, None], y_fit, weights=w_fit)
                model_store[(dataset_name, gene)] = model

                x_pred = np.linspace(float(np.nanmin(x_fit)), float(np.nanmax(x_fit)), int(grid_size))
                y_pred = np.asarray(model.predict(x_pred[:, None]), dtype=float).reshape(-1)
                try:
                    conf = model.confidence_intervals(x_pred[:, None], width=float(confidence_level))
                    lower = np.asarray(conf[:, 0], dtype=float)
                    upper = np.asarray(conf[:, 1], dtype=float)
                except Exception:
                    lower = np.full_like(y_pred, np.nan)
                    upper = np.full_like(y_pred, np.nan)

                peak_time, valley_time = _summarize_trend(x_pred, y_pred)
                stats_dict = getattr(model, "statistics_", {})
                explained_deviance = _pseudo_r2_from_stats(stats_dict)
                p_value = np.nan
                if isinstance(stats_dict, Mapping):
                    pvals = stats_dict.get("p_values")
                    if pvals is not None:
                        pvals = np.asarray(pvals, dtype=float).reshape(-1)
                        if pvals.size > 0:
                            p_value = _safe_float(pvals[-1])

                stats_records.append(
                    {
                        "dataset": dataset_name,
                        "source_dataset": source_dataset_name,
                        "groupby_key": groupby_key,
                        "group": group_name,
                        "gene": gene,
                        "success": True,
                        "error": None,
                        "n_cells": int(x_fit.shape[0]),
                        "exp_ncells": int(_expressed_cells(y_fit)),
                        "peak_time": peak_time,
                        "valley_time": valley_time,
                        "min_pseudotime": float(np.nanmin(x_fit)),
                        "max_pseudotime": float(np.nanmax(x_fit)),
                        "r2": explained_deviance,
                        "explained_deviance": explained_deviance,
                        "p_value": p_value,
                    }
                )
                fitted_records.extend(
                    {
                        "dataset": dataset_name,
                        "source_dataset": source_dataset_name,
                        "groupby_key": groupby_key,
                        "group": group_name,
                        "gene": gene,
                        "pseudotime": float(pt),
                        "fitted": float(fit_val),
                        "lower": float(lo) if np.isfinite(lo) else np.nan,
                        "upper": float(hi) if np.isfinite(hi) else np.nan,
                    }
                    for pt, fit_val, lo, hi in zip(x_pred, y_pred, lower, upper)
                )
                if store_raw:
                    raw_records.extend(
                        {
                            "dataset": dataset_name,
                            "source_dataset": source_dataset_name,
                            "groupby_key": groupby_key,
                            "group": group_name,
                            "gene": gene,
                            "pseudotime": float(pt),
                            "expression": float(expr),
                            "weight": float(weight) if weight is not None and np.isfinite(weight) else np.nan,
                        }
                        for pt, expr, weight in zip(
                            x_fit,
                            y_fit,
                            w_fit if w_fit is not None else [None] * len(x_fit),
                        )
                    )
            except Exception as e:
                stats_records.append(
                    {
                        "dataset": dataset_name,
                        "source_dataset": source_dataset_name,
                        "groupby_key": groupby_key,
                        "group": group_name,
                        "gene": gene,
                        "success": False,
                        "error": str(e),
                    }
                )
            progress.update(1)

    progress.close()

    stats = pd.DataFrame(stats_records)
    if not stats.empty and "p_value" in stats.columns:
        stats["padj"] = np.nan
        for dataset_name, idx in stats.groupby("dataset", sort=False).groups.items():
            dataset_idx = list(idx)
            mask = stats.loc[dataset_idx, "success"].fillna(False) & stats.loc[dataset_idx, "p_value"].notna()
            valid_idx = list(stats.loc[dataset_idx].index[mask])
            if valid_idx:
                from statsmodels.stats.multitest import multipletests

                _, padj, _, _ = multipletests(stats.loc[valid_idx, "p_value"].to_numpy(dtype=float), method="fdr_bh")
                stats.loc[valid_idx, "padj"] = padj
    fitted = pd.DataFrame(fitted_records)
    raw = pd.DataFrame(raw_records) if store_raw else None
    if not include_source_dataset:
        for df in (stats, fitted, raw):
            if df is not None and "source_dataset" in df.columns:
                df.drop(columns=["source_dataset"], inplace=True)

    result = DynamicFeaturesResult(
        stats=stats,
        fitted=fitted,
        raw=raw,
        models=model_store,
        config={
            "pseudotime": pseudotime,
            "groupby": groupby,
            "groups": None if groups is None else list(groups) if not isinstance(groups, Mapping) else dict(groups),
            "layer": layer,
            "use_raw": use_raw,
            "distribution": distribution,
            "link": link,
            "n_splines": n_splines,
            "spline_order": spline_order,
            "grid_size": grid_size,
            "confidence_level": confidence_level,
        },
    )

    if isinstance(data, AnnData) and key_added is not None:
        stats_store = _prepare_table_for_uns(stats)
        fitted_store = _prepare_table_for_uns(fitted)
        raw_store = None if raw is None else _prepare_table_for_uns(raw)
        data.uns[key_added] = {
            "stats": stats_store,
            "fitted": fitted_store,
            "raw": raw_store,
            "config": dict(result.config or {}),
        }

    if verbose:
        success_count = int(stats["success"].fillna(False).sum()) if not stats.empty and "success" in stats.columns else 0
        print(f"\n{Colors.GREEN}{EMOJI['done']} Dynamic feature analysis completed!{Colors.ENDC}")
        print(f"   {Colors.GREEN}✓ Successful fits: {Colors.BOLD}{success_count}/{len(stats)}{Colors.ENDC}")
        print(f"   {Colors.GREEN}✓ Fitted rows: {Colors.BOLD}{len(fitted)}{Colors.ENDC}")
        if store_raw and raw is not None:
            print(f"   {Colors.GREEN}✓ Raw observations stored: {Colors.BOLD}{len(raw)}{Colors.ENDC}")

    return result
