r"""``pyMetabo`` — AnnData-native lifecycle class for metabolomics QC/stats.

Same lifecycle as ``omicverse.pp._scdblfinder.ScDblFinder`` and
``omicverse.single.Milo`` — a class that holds the AnnData, exposes
chainable methods for each pipeline stage, and stashes intermediate
artifacts on ``self`` for later inspection/plotting.

Usage
-----
>>> from omicverse.metabol import pyMetabo, read_metaboanalyst
>>> # group_col is required — pass the factor column name from your CSV
>>> adata = read_metaboanalyst("human_cachexia.csv", group_col="Muscle loss")
>>> m = pyMetabo(adata)
>>> (m.impute(method="qrilc", seed=0)
...    .normalize(method="pqn")
...    .transform(method="log")
...    .differential(method="welch_t", log_transformed=True)
...    .transform(method="pareto", stash_raw=False)
...    .plsda(n_components=2))
>>> m.deg_table.head()
>>> m.plsda_result.to_vip_table(m.adata.var_names).head()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from . import _impute as _imp
from . import _norm as _norm
from . import _plsda as _pls
from . import _qc as _qc_mod
from . import _stats as _stats_mod
from . import _transform as _tf


@dataclass
class pyMetabo:
    """Lifecycle class for a metabolomics analysis.

    Attributes populated as the pipeline runs
    ------------------------------------------
    adata : the current state of the AnnData (updated in place on each call)
    raw   : the original AnnData handed in at construction (never mutated)
    deg_table : DataFrame returned by ``differential()``
    plsda_result : PLSDAResult from ``plsda()`` / ``opls_da()``
    """

    adata: AnnData
    random_state: int = 0
    raw: AnnData = field(init=False)
    deg_table: Optional[pd.DataFrame] = field(default=None, init=False)
    plsda_result: Optional[_pls.PLSDAResult] = field(default=None, init=False)

    def __post_init__(self):
        # Freeze the input for provenance; we always operate on self.adata
        self.raw = self.adata.copy()

    # ------------------------------------------------------------------
    # preprocessing stages
    # ------------------------------------------------------------------
    def cv_filter(self, *, qc_mask, cv_threshold: float = 0.30) -> "pyMetabo":
        self.adata = _qc_mod.cv_filter(self.adata, qc_mask=qc_mask,
                                       cv_threshold=cv_threshold)
        return self

    def drift_correct(self, *, injection_order, qc_mask, frac: float = 0.5) -> "pyMetabo":
        self.adata = _qc_mod.drift_correct(
            self.adata, injection_order=injection_order,
            qc_mask=qc_mask, frac=frac,
        )
        return self

    def blank_filter(self, *, blank_mask, ratio: float = 3.0) -> "pyMetabo":
        self.adata = _qc_mod.blank_filter(self.adata, blank_mask=blank_mask, ratio=ratio)
        return self

    def impute(self, *, method: str = "qrilc", seed: Optional[int] = None,
               **kwargs) -> "pyMetabo":
        # Default to the class's own random_state so the chainable pipeline
        # is reproducible under a single seed.
        if seed is None:
            seed = self.random_state
        self.adata = _imp.impute(self.adata, method=method, seed=seed, **kwargs)
        return self

    def normalize(self, *, method: str = "pqn", **kwargs) -> "pyMetabo":
        self.adata = _norm.normalize(self.adata, method=method, **kwargs)
        return self

    def transform(self, *, method: str = "log", **kwargs) -> "pyMetabo":
        self.adata = _tf.transform(self.adata, method=method, **kwargs)
        return self

    # ------------------------------------------------------------------
    # analysis stages
    # ------------------------------------------------------------------
    def differential(
        self, *,
        group_col: str = "group", group_a: Optional[str] = None,
        group_b: Optional[str] = None, method: str = "welch_t",
        log_transformed: bool = True,
    ) -> "pyMetabo":
        self.deg_table = _stats_mod.differential(
            self.adata, group_col=group_col, group_a=group_a, group_b=group_b,
            method=method, log_transformed=log_transformed,
        )
        return self

    def plsda(self, *, n_components: int = 2, group_col: str = "group",
              group_a: Optional[str] = None, group_b: Optional[str] = None,
              scale: bool = False) -> "pyMetabo":
        self.plsda_result = _pls.plsda(
            self.adata, group_col=group_col, group_a=group_a, group_b=group_b,
            n_components=n_components, scale=scale,
        )
        return self

    def opls_da(self, *, n_ortho: int = 1, group_col: str = "group",
                group_a: Optional[str] = None, group_b: Optional[str] = None,
                scale: bool = False) -> "pyMetabo":
        self.plsda_result = _pls.opls_da(
            self.adata, group_col=group_col, group_a=group_a, group_b=group_b,
            n_ortho=n_ortho, scale=scale,
        )
        return self

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def vip_table(self) -> pd.DataFrame:
        if self.plsda_result is None:
            raise RuntimeError("call .plsda() or .opls_da() first")
        return self.plsda_result.to_vip_table(self.adata.var_names)

    def significant_metabolites(
        self, *, padj_thresh: float = 0.05, log2fc_thresh: float = 1.0
    ) -> pd.DataFrame:
        if self.deg_table is None:
            raise RuntimeError("call .differential() first")
        d = self.deg_table
        return d[(d["padj"] < padj_thresh) & (d["log2fc"].abs() >= log2fc_thresh)]
