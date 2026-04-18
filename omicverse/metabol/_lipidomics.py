r"""Lipidomics-specific helpers.

Lipidomics data has more structure than generic metabolomics because
every lipid follows the **LIPID MAPS shorthand notation**:

    PC 34:1        phosphatidylcholine, 34 carbons total, 1 double bond
    TAG 54:3       triacylglycerol, 54C, 3 double bonds
    LPE 18:0       lysophosphatidylethanolamine, 18C, saturated
    Cer d18:1/24:0 ceramide, sphingosine backbone + 24C saturated N-acyl

:func:`parse_lipid` decodes these strings into a small dataclass; then
:func:`aggregate_by_class` rolls up a lipid abundance matrix to
class-level totals (e.g. "total PC", "total TAG") which is the
standard first-pass for lipid-focused analysis.

:func:`lion_enrichment` runs ORA against a LION-like ontology: subsets
of lipid classes and properties (subcellular localization, function,
physical state) against a hit list. We ship a curated compact LION
subset as ``data/lion_subset.json``; users who need the full LION
should swap in the upstream JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from ._utils import bh_fdr as _bh_fdr


_DATA_DIR = Path(__file__).parent / "data"
_LION_PATH = _DATA_DIR / "lion_subset.json"


@dataclass
class LipidIdentity:
    """Parsed LIPID MAPS shorthand — the sum-composition level."""

    lipid_class: str         # "PC", "TAG", "Cer", "SM", "LPC", ...
    total_carbons: int       # sum of acyl chain carbons (for Cer: sphingosine + N-acyl)
    total_db: int            # total double bonds
    backbone: Optional[str] = None   # "d18:1" for Cer, None otherwise
    raw: str = ""            # original string

    def is_saturated(self) -> bool:
        return self.total_db == 0

    def is_polyunsaturated(self, threshold: int = 2) -> bool:
        return self.total_db >= threshold


# Classes the regex recognizes — extend if you have unusual species
LIPID_CLASSES = (
    "PC", "PE", "PS", "PG", "PI", "PA",
    "LPC", "LPE", "LPS", "LPG", "LPI", "LPA",
    "SM", "Cer", "GlcCer", "LacCer",
    "TAG", "TG", "DAG", "DG", "MAG", "MG",
    "CE", "FA", "Chol", "BMP",
    "Hex2Cer", "Hex3Cer",
)
_CLASS_ALT = "|".join(sorted(LIPID_CLASSES, key=len, reverse=True))
# Match e.g. "PC 34:1", "LPE 18:0", "Cer d18:1/24:0", "TAG 54:3;O"
_PATTERN = re.compile(
    rf"^(?P<klass>{_CLASS_ALT})\s*(?:(?P<backbone>[dmt]\d+:\d+)\/)?(?P<carbons>\d+):(?P<db>\d+)",
    re.IGNORECASE,
)


def parse_lipid(name: str) -> Optional[LipidIdentity]:
    """Parse a LIPID MAPS-shorthand lipid name.

    Returns ``None`` if the name doesn't match any recognized pattern
    (so the caller can filter or annotate as "not a lipid").
    """
    match = _PATTERN.match(str(name).strip())
    if not match:
        return None
    return LipidIdentity(
        lipid_class=match.group("klass").upper(),
        total_carbons=int(match.group("carbons")),
        total_db=int(match.group("db")),
        backbone=match.group("backbone"),
        raw=name,
    )


def annotate_lipids(adata: AnnData, *, feature_names: Optional[Iterable[str]] = None) -> AnnData:
    """Parse each ``var_name`` as a lipid and add ``lipid_class`` /
    ``total_carbons`` / ``total_db`` columns to ``adata.var``.

    Returns a *copy* of ``adata`` — existing columns are preserved.
    Unparseable names get ``lipid_class = NaN``.
    """
    out = adata.copy()
    names = list(feature_names) if feature_names is not None else list(out.var_names)
    classes, carbons, dbs, bbones = [], [], [], []
    for n in names:
        lid = parse_lipid(n)
        if lid is None:
            classes.append(None); carbons.append(np.nan)
            dbs.append(np.nan); bbones.append(None)
        else:
            classes.append(lid.lipid_class); carbons.append(lid.total_carbons)
            dbs.append(lid.total_db); bbones.append(lid.backbone)
    out.var["lipid_class"] = classes
    out.var["total_carbons"] = carbons
    out.var["total_db"] = dbs
    out.var["lipid_backbone"] = bbones
    return out


def aggregate_by_class(adata: AnnData, *, agg: str = "sum") -> AnnData:
    """Collapse the matrix to class-level totals.

    ``adata.var['lipid_class']`` must already exist (run ``annotate_lipids``
    first). Returns a new AnnData with ``n_vars = n_lipid_classes`` and
    per-sample class totals in ``.X``. Handy for quick-look class-level
    QC and for some regression models.
    """
    if "lipid_class" not in adata.var.columns:
        raise KeyError(
            "adata.var has no lipid_class column — call annotate_lipids() first"
        )
    classes = adata.var["lipid_class"].values
    unique = pd.unique(pd.Series(classes).dropna()).tolist()
    if not unique:
        raise ValueError("No lipid species recognized — check var_names format.")

    X_agg = np.zeros((adata.n_obs, len(unique)), dtype=np.float64)
    for j, cls in enumerate(unique):
        cols = np.where(classes == cls)[0]
        block = np.asarray(adata.X[:, cols], dtype=np.float64)
        if agg == "sum":
            X_agg[:, j] = np.nansum(block, axis=1)
        elif agg == "mean":
            X_agg[:, j] = np.nanmean(block, axis=1)
        elif agg == "median":
            X_agg[:, j] = np.nanmedian(block, axis=1)
        else:
            raise ValueError(f"unknown agg={agg!r} (use sum/mean/median)")

    new_var = pd.DataFrame({
        "n_species": [int((classes == c).sum()) for c in unique],
    }, index=unique)
    return AnnData(X=X_agg, obs=adata.obs.copy(), var=new_var)


def _load_lion_subset() -> dict[str, dict]:
    """LION-inspired compact ontology (shipped in data/lion_subset.json).

    For the full ~730-term LION ontology, call
    ``ov.metabol.fetch_lion_associations()`` and pass the result to
    :func:`lion_enrichment` via the ``ontology=`` argument.
    """
    if not _LION_PATH.exists():
        # Fallback to a hard-coded minimal set so tests don't break if the
        # file is missing from a partial install.
        return _FALLBACK_LION
    with open(_LION_PATH) as f:
        return json.load(f)


def lion_enrichment(
    hits: Iterable[str],
    background: Iterable[str],
    *,
    ontology: Optional[dict[str, dict]] = None,
    min_size: int = 3,
) -> pd.DataFrame:
    """LION-style over-representation for lipid classes / properties.

    Parameters
    ----------
    hits
        Lipid names in LIPID MAPS shorthand (e.g. ``['PC 34:1', 'TAG 54:3', ...]``).
    background
        All tested lipid names.
    ontology
        Dict of ``{term_name: {"members": [lipid_class, ...], "category": ...}}``.
        If ``None``, the local LION subset is used.
    """
    ont = ontology if ontology is not None else _load_lion_subset()

    hit_classes = [p.lipid_class for p in (parse_lipid(h) for h in hits) if p]
    bg_classes = [p.lipid_class for p in (parse_lipid(b) for b in background) if p]
    hit_set = set(hit_classes)
    bg_set = set(bg_classes)

    rows = []
    for term, info in ont.items():
        members = set(info["members"])
        overlap_set = hit_set & members
        if len(members & bg_set) < min_size:
            continue
        a = len(overlap_set)
        b = len(hit_set - members)
        c = len((members & bg_set) - hit_set)
        d = len(bg_set - hit_set - members)
        if a == 0:
            continue
        try:
            odds, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        except ValueError:
            continue
        rows.append({
            "term": term,
            "category": info.get("category", ""),
            "overlap": a,
            "set_size": len(members & bg_set),
            "odds_ratio": odds,
            "pvalue": pvalue,
            "hit_members": ";".join(sorted(overlap_set)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["padj"] = _bh_fdr(out["pvalue"].to_numpy())
    return out.sort_values("pvalue").reset_index(drop=True)


# Fallback compact ontology if data/lion_subset.json is missing
_FALLBACK_LION = {
    "Glycerophospholipids": {
        "category": "lipid_class",
        "members": ["PC", "PE", "PS", "PG", "PI", "PA", "BMP",
                    "LPC", "LPE", "LPS", "LPG", "LPI", "LPA"],
    },
    "Lysophospholipids": {
        "category": "lipid_class",
        "members": ["LPC", "LPE", "LPS", "LPG", "LPI", "LPA"],
    },
    "Sphingolipids": {
        "category": "lipid_class",
        "members": ["SM", "Cer", "GlcCer", "LacCer", "Hex2Cer", "Hex3Cer"],
    },
    "Neutral lipids": {
        "category": "lipid_class",
        "members": ["TAG", "TG", "DAG", "DG", "MAG", "MG", "CE"],
    },
    "Membrane constituents": {
        "category": "function",
        "members": ["PC", "PE", "PS", "PI", "SM", "Cer"],
    },
    "Energy storage": {
        "category": "function",
        "members": ["TAG", "TG", "CE"],
    },
    "Signaling lipids": {
        "category": "function",
        "members": ["PI", "PA", "DAG", "DG", "Cer", "LPC", "LPA", "LPE"],
    },
}
