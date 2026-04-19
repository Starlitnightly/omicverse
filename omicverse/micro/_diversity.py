"""Alpha + beta diversity for microbiome AnnData.

Uses ``scikit-bio`` for all single-sample / pairwise metrics; the ``unifrac``
package (``pip install unifrac``) is tapped when ``metric='unifrac'`` and a
phylogenetic tree is supplied.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover
    raise ImportError("anndata is required for ov.micro._diversity") from exc

from .._registry import register_function
from ._utils import dense as _dense, rarefy_counts as _rarefy_counts


# ``scikit-bio`` canonical metric names
ALPHA_METRICS = (
    "shannon",
    "simpson",
    "observed_otus",   # aka Observed ASVs
    "chao1",
    "faith_pd",
    "pielou_e",
    "gini_index",
    "robbins",
)

BETA_METRICS = (
    "braycurtis",
    "jaccard",
    "aitchison",
    "canberra",
    "euclidean",
    "unifrac",
    "unweighted_unifrac",
    "weighted_unifrac",
    "weighted_normalized_unifrac",
)


def _build_tree_file(adata: "ad.AnnData") -> Optional[str]:
    """Write ``uns['tree']`` (newick string) to a temp file for unifrac package."""
    tree = adata.uns.get("tree") if hasattr(adata, "uns") else None
    if not tree:
        return None
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".nwk", delete=False, prefix="omicverse_micro_tree_"
    )
    tmp.write(tree)
    tmp.close()
    return tmp.name


# ----------------------------------------------------------------------------
# Alpha
# ----------------------------------------------------------------------------


@register_function(
    aliases=["alpha_diversity", "Alpha", "within_sample_diversity"],
    category="microbiome",
    description="Alpha diversity: per-sample richness / evenness metrics via scikit-bio.",
    examples=[
        "ov.micro.Alpha(adata).run(metrics=['shannon', 'observed_otus'])",
    ],
    related=["micro.Beta", "micro.Ordinate"],
)
class Alpha:
    """Compute and store per-sample alpha-diversity metrics on ``adata.obs``.

    Parameters
    ----------
    adata
        Samples × features AnnData with int counts. The class does NOT
        modify ``X``; pass ``rarefy_depth`` if you want rarefaction applied
        only for the diversity calculation.
    rarefy_depth
        If set, subsample to this depth (without replacement) before
        computing each metric. ``None`` → use raw counts.
    seed
        Random seed for rarefaction.
    """

    def __init__(
        self,
        adata: "ad.AnnData",
        rarefy_depth: Optional[int] = None,
        seed: int = 0,
    ):
        self.adata = adata
        self.rarefy_depth = rarefy_depth
        self.seed = seed
        self.result_: Optional[pd.DataFrame] = None

    def _counts(self) -> np.ndarray:
        return _rarefy_counts(_dense(self.adata.X), self.rarefy_depth, self.seed)

    def run(
        self,
        metrics: Union[str, Sequence[str]] = ("shannon", "observed_otus"),
        write_to_obs: bool = True,
        tree_key: str = "tree",
    ) -> pd.DataFrame:
        """Compute the requested alpha metrics.

        Returns a DataFrame indexed by sample with one column per metric.
        By default the result is also merged into ``adata.obs`` with the
        same column names.
        """
        try:
            from skbio.diversity import alpha_diversity
        except ImportError as exc:
            raise ImportError(
                "ov.micro.Alpha requires scikit-bio (pip install scikit-bio)."
            ) from exc

        metrics = [metrics] if isinstance(metrics, str) else list(metrics)
        counts = self._counts()
        ids = list(self.adata.obs_names)

        results: dict[str, pd.Series] = {}
        tree_file = None
        for m in metrics:
            if m not in ALPHA_METRICS:
                # skbio accepts other names too — just pass-through
                pass
            if m == "faith_pd":
                tree_file = tree_file or _build_tree_file(self.adata)
                if not tree_file:
                    raise ValueError(
                        "faith_pd requires a phylogenetic tree stored at "
                        f"adata.uns[{tree_key!r}] as a newick string."
                    )
                series = alpha_diversity(
                    "faith_pd",
                    counts,
                    ids=ids,
                    otu_ids=list(self.adata.var_names),
                    tree=tree_file,
                )
            else:
                series = alpha_diversity(m, counts, ids=ids)
            results[m] = series.reindex(ids)

        df = pd.DataFrame(results)
        self.result_ = df
        if write_to_obs:
            for col in df.columns:
                self.adata.obs[col] = df[col].reindex(self.adata.obs_names).values
        return df

    def shannon(self) -> pd.Series:
        return self.run("shannon")["shannon"]

    def observed(self) -> pd.Series:
        return self.run("observed_otus")["observed_otus"]


# ----------------------------------------------------------------------------
# Beta
# ----------------------------------------------------------------------------


@register_function(
    aliases=["beta_diversity", "Beta", "pairwise_distance"],
    category="microbiome",
    description="Beta diversity: sample × sample dissimilarity matrices "
                "(Bray-Curtis / Jaccard / Aitchison / UniFrac) via scikit-bio.",
    examples=[
        "ov.micro.Beta(adata).run(metric='braycurtis', rarefy=True)",
    ],
    related=["micro.Alpha", "micro.Ordinate"],
)
class Beta:
    """Compute sample × sample distance matrices.

    Parameters
    ----------
    adata
        Samples × features AnnData with int counts.
    rarefy_depth
        If set, subsample to this depth before distance calculation
        (good practice for Bray-Curtis / Jaccard to remove library-size
        effects). ``None`` → use raw counts.
    seed
        Random seed for rarefaction.
    """

    def __init__(
        self,
        adata: "ad.AnnData",
        rarefy_depth: Optional[int] = None,
        seed: int = 0,
    ):
        self.adata = adata
        self.rarefy_depth = rarefy_depth
        self.seed = seed
        self.dm_: dict[str, "pd.DataFrame"] = {}

    def _counts(self, rarefy_depth: Optional[int]) -> np.ndarray:
        return _rarefy_counts(_dense(self.adata.X), rarefy_depth, self.seed)

    def run(
        self,
        metric: str = "braycurtis",
        rarefy: Optional[bool] = None,
        tree_key: str = "tree",
        write_to_obsp: bool = True,
    ) -> pd.DataFrame:
        """Compute the distance matrix.

        Parameters
        ----------
        metric
            Any name accepted by ``skbio.diversity.beta_diversity`` plus
            ``'unifrac'`` (unweighted) / ``'weighted_unifrac'`` /
            ``'weighted_normalized_unifrac'``. UniFrac variants require a
            tree in ``adata.uns[tree_key]`` (newick string) AND
            ``adata.var_names`` matching tree tips.
        rarefy
            Force-enable / disable rarefaction for this call (overrides the
            constructor default).
        write_to_obsp
            If True, store the result in ``adata.obsp[metric]`` as a
            (n_obs, n_obs) square matrix for compatibility with scanpy's
            downstream tools.
        """
        if metric not in BETA_METRICS:
            # still pass-through to skbio
            pass
        # Resolve the rarefaction depth for THIS call without mutating self.
        call_depth: Optional[int] = self.rarefy_depth
        if rarefy is True and call_depth is None:
            X = _dense(self.adata.X)
            call_depth = int(X.sum(axis=1).min())
        elif rarefy is False:
            call_depth = None
        counts = self._counts(call_depth)
        ids = list(self.adata.obs_names)

        if "unifrac" in metric.lower():
            dm = self._beta_unifrac(counts, metric, tree_key, call_depth)
        else:
            try:
                from skbio.diversity import beta_diversity
            except ImportError as exc:
                raise ImportError(
                    "ov.micro.Beta requires scikit-bio (pip install scikit-bio)."
                ) from exc
            dm_skbio = beta_diversity(metric, counts, ids=ids)
            dm = pd.DataFrame(dm_skbio.data, index=ids, columns=ids)

        self.dm_[metric] = dm
        if write_to_obsp:
            self.adata.obsp[metric] = dm.loc[self.adata.obs_names,
                                              self.adata.obs_names].values
        return dm

    def _beta_unifrac(self, counts: np.ndarray, metric: str, tree_key: str,
                      call_depth: Optional[int] = None) -> pd.DataFrame:
        tree = self.adata.uns.get(tree_key)
        if not tree:
            raise ValueError(
                f"UniFrac metrics require a newick tree at adata.uns[{tree_key!r}]."
            )
        try:
            from skbio.diversity import beta_diversity
        except ImportError as exc:
            raise ImportError(
                "Beta requires scikit-bio for UniFrac metrics."
            ) from exc
        tf = _build_tree_file(self.adata)
        ids = list(self.adata.obs_names)
        otu_ids = list(self.adata.var_names)
        dm_skbio = beta_diversity(
            metric, counts, ids=ids, otu_ids=otu_ids, tree=tf, validate=True
        )
        return pd.DataFrame(dm_skbio.data, index=ids, columns=ids)

    def braycurtis(self, rarefy: bool = True) -> pd.DataFrame:
        return self.run("braycurtis", rarefy=rarefy)
