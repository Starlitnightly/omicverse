"""Ordination for microbiome AnnData.

PCoA (principal coordinates) and NMDS operate on the beta-diversity distance
matrices produced by :class:`omicverse.micro.Beta`. Results are stored in
``adata.obsm[<dist>_pcoa]`` / ``adata.obsm[<dist>_nmds]`` so they slot into
``scanpy.pl`` or matplotlib directly.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover
    raise ImportError("anndata is required for ov.micro._ord") from exc

from .._registry import register_function


@register_function(
    aliases=["Ordinate", "ordination", "pcoa"],
    category="microbiome",
    description="Ordination (PCoA / NMDS) on a beta-diversity distance matrix.",
    examples=[
        "ov.micro.Ordinate(adata, dist_key='braycurtis').pcoa(n=3)",
    ],
    related=["micro.Beta"],
)
class Ordinate:
    """Reduce a sample × sample distance matrix to 2-D / 3-D coords.

    Parameters
    ----------
    adata
        AnnData with a distance matrix already computed by
        :meth:`Beta.run` (stored in ``adata.obsp[dist_key]``).
    dist_key
        Key into ``adata.obsp``. Defaults to ``'braycurtis'``.
    """

    def __init__(self, adata: "ad.AnnData", dist_key: str = "braycurtis"):
        self.adata = adata
        self.dist_key = dist_key
        if dist_key not in adata.obsp:
            raise KeyError(
                f"adata.obsp does not contain {dist_key!r}. "
                "Compute it first, e.g. ov.micro.Beta(adata).run(metric='braycurtis')."
            )
        self.D = np.asarray(adata.obsp[dist_key])
        self.result_: dict[str, np.ndarray] = {}
        self.proportion_explained_: Optional[np.ndarray] = None

    # ----- PCoA -----

    @register_function(
        aliases=["ordinate.pcoa", "pcoa_run", "principal_coordinates"],
        category="microbiome",
        description="Principal Coordinates Analysis on a sample × sample distance matrix; writes coords into adata.obsm.",
        examples=[
            "ov.micro.Ordinate(adata, dist_key='braycurtis').pcoa(n=3)",
        ],
        related=["micro.Beta", "micro.Ordinate"],
    )
    def pcoa(self, n: int = 3, write_to_obsm: bool = True) -> pd.DataFrame:
        """Principal coordinates analysis.

        Stores coords in ``adata.obsm[f'{dist_key}_pcoa']`` and the first
        `n` eigenvalues' proportion explained in
        ``adata.uns['micro'][f'{dist_key}_pcoa_var']``.
        """
        try:
            from skbio.stats.ordination import pcoa as _skbio_pcoa
            from skbio import DistanceMatrix
        except ImportError as exc:
            raise ImportError(
                "Ordinate.pcoa requires scikit-bio."
            ) from exc
        dm = DistanceMatrix(self.D, ids=list(self.adata.obs_names))
        res = _skbio_pcoa(dm, number_of_dimensions=n)
        coords = res.samples.values
        pct = np.asarray(res.proportion_explained)
        self.result_["pcoa"] = coords
        self.proportion_explained_ = pct
        key = f"{self.dist_key}_pcoa"
        if write_to_obsm:
            self.adata.obsm[key] = coords
            self.adata.uns.setdefault("micro", {})[f"{key}_var"] = pct[:n]
        return pd.DataFrame(coords,
                            index=self.adata.obs_names,
                            columns=[f"PC{i+1}" for i in range(coords.shape[1])])

    # ----- NMDS (non-metric) -----

    @register_function(
        aliases=["ordinate.nmds", "nmds_run", "non_metric_mds"],
        category="microbiome",
        description="Non-metric multidimensional scaling on a sample × sample distance matrix (via sklearn); writes coords into adata.obsm.",
        examples=[
            "ov.micro.Ordinate(adata, dist_key='braycurtis').nmds(n=2)",
        ],
        related=["micro.Ordinate"],
    )
    def nmds(self, n: int = 2, random_state: int = 0,
             write_to_obsm: bool = True) -> pd.DataFrame:
        """Non-metric multi-dimensional scaling (via sklearn)."""
        try:
            from sklearn.manifold import MDS
        except ImportError as exc:
            raise ImportError("Ordinate.nmds requires scikit-learn.") from exc
        mds = MDS(n_components=n, dissimilarity="precomputed",
                  random_state=random_state, n_init=4,
                  normalized_stress="auto")
        coords = mds.fit_transform(self.D)
        self.result_["nmds"] = coords
        key = f"{self.dist_key}_nmds"
        if write_to_obsm:
            self.adata.obsm[key] = coords
            self.adata.uns.setdefault("micro", {})[f"{key}_stress"] = mds.stress_
        return pd.DataFrame(coords,
                            index=self.adata.obs_names,
                            columns=[f"NMDS{i+1}" for i in range(n)])

    def proportion_explained(self) -> Optional[np.ndarray]:
        """Eigenvalue proportions from the most recent PCoA call."""
        return self.proportion_explained_
