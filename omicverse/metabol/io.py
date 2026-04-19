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

from .._registry import register_function


@register_function(
    aliases=["代谢组导入", "read_metaboanalyst", "metaboanalyst_csv"],
    category="metabolomics",
    description=(
        "Load a MetaboAnalyst-format peak CSV into AnnData (samples × metabolites). "
        "group_col is required — factor column name differs per dataset."
    ),
    examples=[
        "ov.metabol.read_metaboanalyst('human_cachexia.csv', group_col='Muscle loss')",
    ],
    related=["metabol.read_wide", "metabol.read_lcms", "metabol.pyMetabo"],
)
def read_metaboanalyst(
    path: str | Path,
    *,
    group_col: str,
    sample_col: Optional[str] = None,
    transpose: bool = False,
) -> AnnData:
    """Load a MetaboAnalyst-format CSV into AnnData.

    MetaboAnalyst's canonical bulk-upload layout is::

        Patient ID,Muscle loss,Metabolite1,Metabolite2,...
        PIF_1,cachexic,40.85,62.23,...
        PIF_2,control,29.46,58.13,...

    — samples in rows, one factor (group) column, rest are metabolite
    concentrations. ``group_col`` is **required** (no default) because
    every MetaboAnalyst dataset uses a different column name
    (``"Muscle loss"`` for the Eisner-2010 cachexia demo, ``"Label"`` for
    most others). Pass the exact column header from your CSV.

    Parameters
    ----------
    path
        CSV path or URL.
    group_col
        Name of the factor column holding case/control labels. Required;
        copied to ``adata.obs['group']``.
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
            f"group_col={group_col!r} is not a column in the CSV. "
            f"Available columns (first 8): {list(df.columns[:8])}. "
            f"Pass the header name of the factor column, e.g. "
            f"read_metaboanalyst(path, group_col='{df.columns[1] if len(df.columns) > 1 else 'group'}')."
        )
    obs = df[[sample_col_eff, group_col]].copy()
    obs.columns = ["sample", "group"]
    obs = obs.set_index("sample")

    metabolite_cols = [c for c in df.columns if c not in (sample_col_eff, group_col)]
    X = df[metabolite_cols].to_numpy(dtype=np.float64)
    var = pd.DataFrame(index=metabolite_cols)
    return AnnData(X=X, obs=obs, var=var)


@register_function(
    aliases=["read_wide", "代谢组宽表"],
    category="metabolomics",
    description=(
        "Load a generic wide (samples × metabolites) table into AnnData. "
        "Use when you have a plain TSV / CSV without the MetaboAnalyst convention."
    ),
    examples=["ov.metabol.read_wide('peak_table.tsv', group_col='condition')"],
    related=["metabol.read_metaboanalyst", "metabol.read_lcms"],
)
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


@register_function(
    aliases=["read_lcms", "untargeted_lcms", "非靶向LC-MS"],
    category="metabolomics",
    description=(
        "Load an LC-MS peak table with m/z_RT feature IDs. Parses feature "
        "identifiers into numeric var['m_z'] and var['rt'] so mummichog "
        "can consume the AnnData directly. Handles both MetaboAnalyst "
        "(features-in-rows with embedded Label row) and generic layouts."
    ),
    examples=[
        "ov.metabol.read_lcms('malaria_feature_table.csv', "
        "feature_id_sep='__', label_row='Label', transpose=True)",
    ],
    related=["metabol.annotate_peaks", "metabol.mummichog_basic"],
)
def read_lcms(
    path: str | Path,
    *,
    feature_id_sep: str = "/",
    sample_col: Optional[str] = None,
    group_col: Optional[str] = None,
    label_row: Optional[str] = None,
    transpose: bool = True,
) -> AnnData:
    """Load an LC-MS peak table with ``m/z``/``RT`` feature IDs into AnnData.

    Handles MetaboAnalyst's LC-MS demo layout (peaks-in-rows with a
    dedicated "Label" row above the data), as well as plain wide tables.
    Feature IDs of the form ``<m/z><sep><RT>`` (e.g. ``"200.1/2926"`` or
    ``"85.065__24.64"``) are parsed into numeric ``var['m_z']`` /
    ``var['rt']`` columns so :func:`mummichog_basic` can consume the
    AnnData directly.

    Parameters
    ----------
    feature_id_sep
        Separator between m/z and RT inside each feature ID. MetaboAnalyst
        uses ``"/"`` for the small demo and ``"__"`` for the malaria
        table; XCMS exports sometimes use ``"_"``.
    label_row
        When the raw file has features in rows, the first "feature" is
        often a dedicated *group-label* row (e.g. MetaboAnalyst's
        ``"Label"`` row with values like ``"Naive"`` / ``"Semi_immue"``).
        Pass the label of that row here and we'll lift it into
        ``adata.obs['group']`` before treating the rest as features.
    transpose
        Default True (peaks-in-rows → samples-in-rows, AnnData orientation).
    sample_col, group_col
        Only used when the table is already in samples-in-rows layout;
        both default to the first column / the first non-sample column.
    """
    df = pd.read_csv(path)
    label_values = None
    if transpose:
        feature_id = df.columns[0]
        # Extract the label row *before* transposing so the rest of the
        # table stays purely numeric (otherwise pandas infers `object`
        # dtype for the whole frame and down-stream `.astype(float)`
        # fails).
        if label_row is not None:
            matches = df[df[feature_id] == label_row]
            if matches.empty:
                raise KeyError(
                    f"label_row={label_row!r} not found in first column"
                )
            label_values = matches.iloc[0, 1:].to_numpy()
            df = df[df[feature_id] != label_row].copy()
        df = df.set_index(feature_id).T.reset_index().rename(columns={"index": "sample"})
    sample_col_eff = sample_col or df.columns[0]
    obs_cols = [c for c in (group_col,) if c]
    obs = df[[sample_col_eff, *obs_cols]].copy().set_index(sample_col_eff)
    if label_values is not None:
        obs["group"] = label_values
    met_cols = [c for c in df.columns if c != sample_col_eff and c not in obs_cols]
    X = df[met_cols].to_numpy(dtype=np.float64)
    var = pd.DataFrame(index=met_cols)
    # Parse m/z and RT from the feature IDs (mummichog wants numeric columns)
    parts = var.index.to_series().str.split(feature_id_sep, n=1, expand=True)
    if parts.shape[1] == 2:
        try:
            var["m_z"] = parts.iloc[:, 0].astype(float).to_numpy()
            var["rt"] = parts.iloc[:, 1].astype(float).to_numpy()
        except (ValueError, TypeError):
            # Non-numeric feature IDs — leave as-is so the caller can
            # annotate manually.
            pass
    return AnnData(X=X, obs=obs, var=var)
