r"""I/O helpers for metabolomics data files.

Loads peak tables into AnnData with ``obs = samples``, ``var = metabolites``.
The assumption is that upstream peak picking / alignment (XCMS, MZmine,
MS-DIAL, OpenMS, Compound Discoverer) has already been done — this module
only handles the peak-table stage.

Supported formats
-----------------
* **MetaboAnalyst CSV** (``read_metaboanalyst``) — the most common bulk
  upload format: samples in rows, one factor column (group label),
  remaining columns are metabolite concentrations.
* **Wide TSV** (``read_wide``) — generic ``samples × metabolites`` table.
* **LC-MS feature table** (``read_lcms``) — ``m/z_RT`` feature IDs that
  get parsed into ``var['m_z']`` / ``var['rt']`` columns for mummichog
  downstream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData


def read_metaboanalyst(
    path: str | Path,
    *,
    group_col: str = "Muscle loss",
    sample_col: Optional[str] = None,
    transpose: bool = False,
) -> AnnData:
    """Load a MetaboAnalyst-format CSV into AnnData.

    MetaboAnalyst's canonical bulk-upload layout is::

        Patient ID,Muscle loss,Metabolite1,Metabolite2,...
        PIF_1,cachexic,40.85,62.23,...
        PIF_2,control,29.46,58.13,...

    — samples in rows, one factor (group) column, rest are metabolite
    concentrations. The default ``group_col='Muscle loss'`` matches the
    famous Eisner-2010 ``human_cachexia.csv`` demo; pass your own group
    column name otherwise. If ``sample_col`` is None, the first column
    is treated as the sample ID index.

    Parameters
    ----------
    path
        CSV path or URL.
    group_col
        Name of the factor column holding case/control labels. Copied
        to ``adata.obs['group']``.
    sample_col
        Name of the sample-ID column. If ``None``, the first column is
        used (matches the MetaboAnalyst default).
    transpose
        Set to True if your CSV has samples in *columns* and
        metabolites in rows — we'll transpose before wrapping.

    Returns
    -------
    AnnData
        ``adata.X`` is ``n_samples × n_metabolites`` floats; ``obs``
        carries the group label; ``var`` is empty initially.
    """
    df = pd.read_csv(path)
    if transpose:
        df = df.set_index(df.columns[0]).T.reset_index().rename(columns={"index": "sample"})
    sample_col_eff = sample_col or df.columns[0]
    if group_col not in df.columns:
        raise KeyError(
            f"group_col={group_col!r} not in CSV columns: {list(df.columns[:8])}..."
        )
    obs = df[[sample_col_eff, group_col]].copy()
    obs.columns = ["sample", "group"]
    obs = obs.set_index("sample")

    metabolite_cols = [c for c in df.columns if c not in (sample_col_eff, group_col)]
    X = df[metabolite_cols].to_numpy(dtype=np.float64)
    var = pd.DataFrame(index=metabolite_cols)
    return AnnData(X=X, obs=obs, var=var)


def read_wide(
    path: str | Path,
    *,
    sep: str = "\t",
    sample_col: Optional[str] = None,
    group_col: Optional[str] = None,
) -> AnnData:
    """Load a generic wide (samples × metabolites) table into AnnData."""
    df = pd.read_csv(path, sep=sep)
    sample_col_eff = sample_col or df.columns[0]
    obs_cols = [c for c in (group_col,) if c]
    obs = df[[sample_col_eff, *obs_cols]].copy().set_index(sample_col_eff)
    met_cols = [c for c in df.columns if c != sample_col_eff and c not in obs_cols]
    X = df[met_cols].to_numpy(dtype=np.float64)
    return AnnData(X=X, obs=obs, var=pd.DataFrame(index=met_cols))


def read_lcms(
    path: str | Path,
    *,
    feature_id_sep: str = "/",
    sample_col: Optional[str] = None,
    group_col: Optional[str] = None,
    transpose: bool = True,
) -> AnnData:
    """Load an LC-MS peak table with ``m/z/RT`` feature IDs.

    MetaboAnalyst's LC-MS demo format has peaks in rows and feature IDs
    like ``"200.1/2926"`` (m/z slash retention-time-seconds). We split
    the IDs into numeric ``var['m_z']`` / ``var['rt']`` columns that
    ``pyMummichog`` can consume directly.

    Parameters
    ----------
    transpose
        Default True (peaks-in-rows → samples-in-rows, matching AnnData).
    """
    df = pd.read_csv(path)
    if transpose:
        feature_id = df.columns[0]
        df = df.set_index(feature_id).T.reset_index().rename(columns={"index": "sample"})
    sample_col_eff = sample_col or df.columns[0]
    obs_cols = [c for c in (group_col,) if c]
    obs = df[[sample_col_eff, *obs_cols]].copy().set_index(sample_col_eff)
    met_cols = [c for c in df.columns if c != sample_col_eff and c not in obs_cols]
    X = df[met_cols].to_numpy(dtype=np.float64)
    var = pd.DataFrame(index=met_cols)
    parsed = var.index.str.split(feature_id_sep, n=1, expand=True)
    if parsed.nlevels == 2:
        try:
            var["m_z"] = parsed.get_level_values(0).astype(float)
            var["rt"] = parsed.get_level_values(1).astype(float)
        except (ValueError, TypeError):
            pass  # Non-numeric feature IDs — leave as-is
    return AnnData(X=X, obs=obs, var=var)
