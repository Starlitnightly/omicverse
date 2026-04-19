"""Differential abundance testing for microbiome AnnData.

Supported backends:
  - 'wilcoxon' — Mann-Whitney U per feature (fast, non-parametric, no R)
  - 'deseq2'   — negative-binomial GLM via pydeseq2 (count-based, reuses
                 ov.bulk.pyDEG's backend)
  - 'ancombc'  — ANCOM-BC2 via ``skbio.stats.composition.ancombc``
                 (skbio >= 0.7.1; native Python, Nearing 2024 benchmark recs)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover
    raise ImportError("anndata is required for ov.micro._da") from exc

from .._registry import register_function
from ._utils import dense as _dense


@register_function(
    aliases=["DA", "differential_abundance"],
    category="microbiome",
    description="Differential abundance testing: Wilcoxon / pyDESeq2 / ANCOM-BC.",
    examples=[
        "ov.micro.DA(adata).wilcoxon(group_key='group', rank='genus')",
    ],
    related=["micro.Alpha", "bulk.pyDEG"],
)
class DA:
    """Per-feature differential abundance across sample groups.

    Parameters
    ----------
    adata
        Samples × features AnnData with int counts in ``X``.
    """

    def __init__(self, adata: "ad.AnnData"):
        self.adata = adata
        self.result_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _features(self, rank: Optional[str]):
        """Return (feature_matrix, feature_labels) — optionally rank-collapsed."""
        X = _dense(self.adata.X).astype(np.int64)
        if rank is None:
            return X, np.asarray(self.adata.var_names)
        if rank not in self.adata.var.columns:
            raise ValueError(
                f"rank {rank!r} not in adata.var columns "
                f"(available: {list(self.adata.var.columns)})"
            )
        labels = self.adata.var[rank].replace({"": "Unassigned"}).fillna("Unassigned").values
        df = pd.DataFrame(X, index=self.adata.obs_names, columns=labels)
        agg = df.T.groupby(level=0).sum().T
        return agg.values.astype(np.int64), np.asarray(agg.columns)

    def _two_group_mask(self, group_key: str,
                        group_a: str, group_b: str):
        if group_key not in self.adata.obs.columns:
            raise ValueError(f"{group_key!r} not in adata.obs.")
        col = self.adata.obs[group_key]
        ma = (col == group_a).values
        mb = (col == group_b).values
        if ma.sum() == 0 or mb.sum() == 0:
            raise ValueError(
                f"group '{group_a}' n={int(ma.sum())}, "
                f"group '{group_b}' n={int(mb.sum())} — both must be >0."
            )
        return ma, mb

    # ------------------------------------------------------------------
    # wilcoxon
    # ------------------------------------------------------------------

    def wilcoxon(
        self,
        group_key: str,
        group_a: Optional[str] = None,
        group_b: Optional[str] = None,
        rank: Optional[str] = None,
        relative: bool = True,
        min_prevalence: float = 0.1,
    ) -> pd.DataFrame:
        """Two-group Mann-Whitney U test per feature.

        Parameters
        ----------
        group_key
            Column in ``adata.obs`` holding group labels.
        group_a, group_b
            Labels of the two groups to compare. If omitted, the first two
            unique values (sorted) are used.
        rank
            If given, collapse to this taxonomic rank first.
        relative
            Test relative abundances (proportions per sample) rather than
            raw counts. Mann-Whitney is scale-invariant so this mostly
            affects the reported fold-change.
        min_prevalence
            Skip features present (>0) in fewer than this fraction of samples.
        """
        from scipy.stats import mannwhitneyu
        counts, features = self._features(rank)

        if group_a is None or group_b is None:
            vals = sorted(self.adata.obs[group_key].dropna().unique().tolist())
            if len(vals) < 2:
                raise ValueError(
                    f"Need at least 2 unique {group_key!r} values; got {vals}"
                )
            group_a = group_a or vals[0]
            group_b = group_b or vals[1]
        ma, mb = self._two_group_mask(group_key, group_a, group_b)

        if relative:
            row_sums = counts.sum(axis=1, keepdims=True).clip(min=1)
            M = counts / row_sums
        else:
            M = counts.astype(np.float64)

        prev = (counts > 0).sum(axis=0) / counts.shape[0]
        keep = prev >= min_prevalence

        rows = []
        for j, name in enumerate(features):
            if not keep[j]:
                continue
            a = M[ma, j]
            b = M[mb, j]
            if a.sum() == 0 and b.sum() == 0:
                continue
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                stat, p = np.nan, np.nan
            ma_mean = a.mean() if len(a) else np.nan
            mb_mean = b.mean() if len(b) else np.nan
            rows.append({
                "feature": name,
                f"mean_{group_a}": ma_mean,
                f"mean_{group_b}": mb_mean,
                f"log2FC({group_b}/{group_a})":
                    np.log2((mb_mean + 1e-12) / (ma_mean + 1e-12)),
                "U_stat": stat,
                "p_value": p,
                "prevalence": prev[j],
            })
        df = pd.DataFrame(rows)
        if len(df):
            df["fdr_bh"] = _bh_fdr(df["p_value"].values)
            df = df.sort_values("p_value")
        self.result_ = df
        self.adata.uns.setdefault("micro", {}).setdefault("da", {})[
            f"wilcoxon_{group_key}_{group_a}_vs_{group_b}_{rank or 'asv'}"
        ] = df
        return df

    # ------------------------------------------------------------------
    # DESeq2 (count-based)
    # ------------------------------------------------------------------

    def deseq2(
        self,
        group_key: str,
        group_a: Optional[str] = None,
        group_b: Optional[str] = None,
        rank: Optional[str] = None,
        min_prevalence: float = 0.1,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Differential abundance via pyDESeq2 (negative-binomial GLM)."""
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
        except ImportError as exc:
            raise ImportError(
                "DA.deseq2 requires pydeseq2 (pip install pydeseq2)."
            ) from exc
        counts, features = self._features(rank)

        if group_a is None or group_b is None:
            vals = sorted(self.adata.obs[group_key].dropna().unique().tolist())
            group_a = group_a or vals[0]
            group_b = group_b or vals[1]
        ma, mb = self._two_group_mask(group_key, group_a, group_b)
        sel = ma | mb

        # prevalence filter (on selected samples)
        sel_counts = counts[sel]
        prev = (sel_counts > 0).sum(axis=0) / sel_counts.shape[0]
        keep = prev >= min_prevalence
        X = sel_counts[:, keep].astype(np.int32)
        feats = features[keep]
        meta_df = pd.DataFrame(
            {group_key: self.adata.obs[group_key].values[sel]},
            index=np.asarray(self.adata.obs_names)[sel],
        )
        # pydeseq2 expects samples × features; matches our layout
        count_df = pd.DataFrame(X, index=meta_df.index, columns=feats)
        dds = DeseqDataSet(
            counts=count_df,
            metadata=meta_df,
            design_factors=group_key,
            quiet=True,
        )
        dds.deseq2()
        ds = DeseqStats(dds, alpha=alpha,
                        contrast=[group_key, group_b, group_a], quiet=True)
        ds.summary()
        df = ds.results_df.reset_index().rename(
            columns={"index": "feature", "baseMean": "base_mean",
                     "log2FoldChange": f"log2FC({group_b}/{group_a})",
                     "lfcSE": "log2FC_se", "pvalue": "p_value",
                     "padj": "fdr_bh"})
        df["prevalence"] = prev[keep]
        df = df.sort_values("p_value")
        self.result_ = df
        self.adata.uns.setdefault("micro", {}).setdefault("da", {})[
            f"deseq2_{group_key}_{group_a}_vs_{group_b}_{rank or 'asv'}"
        ] = df
        return df

    # ------------------------------------------------------------------
    # ANCOM-BC (skbio 0.7+ native)
    # ------------------------------------------------------------------

    def ancombc(
        self,
        group_key: str,
        rank: Optional[str] = None,
        min_prevalence: float = 0.1,
    ) -> pd.DataFrame:
        """ANCOM-BC via ``skbio.stats.composition.ancombc`` (skbio ≥ 0.7.1).

        .. note::
            The scikit-bio ANCOM-BC API is evolving (the 0.7.1 return type
            is a named-tuple-like bundle, not a DataFrame). This wrapper
            expects the 0.7.1 shape with attributes ``lfc / se / W / p / q
            / diff_abn`` indexed by feature and will raise
            :class:`NotImplementedError` on other shapes so the caller
            knows to pin scikit-bio.
        """
        try:
            from skbio.stats.composition import ancombc as _ancombc
        except ImportError as exc:
            raise ImportError(
                "DA.ancombc requires scikit-bio >= 0.7.1 "
                "(pip install 'scikit-bio>=0.7.1')."
            ) from exc
        counts, features = self._features(rank)
        prev = (counts > 0).sum(axis=0) / counts.shape[0]
        keep = prev >= min_prevalence
        X = counts[:, keep]
        feats = features[keep]
        table = pd.DataFrame(X, index=self.adata.obs_names, columns=feats)
        meta = pd.DataFrame(
            {group_key: self.adata.obs[group_key].values},
            index=self.adata.obs_names,
        )
        res = _ancombc(table=table, metadata=meta, formula=group_key)

        try:
            df = pd.DataFrame({
                "lfc":      pd.Series(res.lfc),
                "se":       pd.Series(res.se),
                "W":        pd.Series(res.W),
                "p_value":  pd.Series(res.p),
                "q_value":  pd.Series(res.q),
                "diff_abn": pd.Series(res.diff_abn),
            })
            df.index.name = "feature"
            df = df.reset_index()
        except (AttributeError, TypeError) as exc:
            raise NotImplementedError(
                f"Installed scikit-bio returns an unsupported ANCOM-BC "
                f"shape ({type(res).__name__}). Pin scikit-bio to a "
                f"supported version (>=0.7.1) or open an issue."
            ) from exc

        feat_prev = pd.Series(prev[keep], index=feats)
        df["prevalence"] = df["feature"].map(feat_prev).values

        self.result_ = df
        self.adata.uns.setdefault("micro", {}).setdefault("da", {})[
            f"ancombc_{group_key}_{rank or 'asv'}"
        ] = df
        return df


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (same formula as statsmodels)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0, 1)
    return out
