r"""Cross-reference metabolite names ↔ HMDB / KEGG / ChEBI / LipidMaps IDs.

Two-level strategy:

1. **Local curated subset** — a parquet file shipped with omicverse covering
   the common metabolites in the MetaboAnalyst demo datasets (and the
   KEGG-compound pathway membership needed for ``pyMSEA``). Loaded eagerly
   at import time; fast for tutorial workflows.

2. **Online fallback** via ``bioservices`` (ChEBI, KEGG REST) for anything
   not in the local table. The user must have network access; fetches are
   cached under ``~/.cache/omicverse/metabol/``.

Local table schema (``metabolite_lookup.parquet``)
--------------------------------------------------
    name        str    lower-cased canonical name (matches MetaboAnalyst CSV headers)
    aliases     list   other accepted names (semicolon-joined in the parquet)
    hmdb        str    e.g. "HMDB0000123"
    kegg        str    e.g. "C00186"
    chebi       str    e.g. "CHEBI:16651"
    lipidmaps   str    e.g. "LMFA01010001" (if applicable)
    mw          float  monoisotopic mass (for m/z matching)

Missing values are stored as empty strings (not NaN) for consistent
string handling downstream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


_DATA_DIR = Path(__file__).parent / "data"
_LOOKUP_PATH = _DATA_DIR / "metabolite_lookup.parquet"


def _load_lookup() -> pd.DataFrame:
    if not _LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"lookup parquet missing at {_LOOKUP_PATH}. "
            "Rebuild it with `omicverse.metabol.build_lookup()`."
        )
    return pd.read_parquet(_LOOKUP_PATH)


def normalize_name(s: str) -> str:
    """Lowercase + strip — the canonical form we index the lookup on."""
    return " ".join(str(s).strip().lower().split())


def map_ids(
    names: Iterable[str],
    *,
    targets: tuple[str, ...] = ("hmdb", "kegg", "chebi"),
    allow_online: bool = False,
) -> pd.DataFrame:
    """Resolve a list of metabolite names to external database IDs.

    Parameters
    ----------
    names
        Iterable of metabolite names (e.g. ``adata.var_names``).
    targets
        Which external IDs to resolve — any subset of
        ``("hmdb", "kegg", "chebi", "lipidmaps")``.
    allow_online
        If True, fall back to ``bioservices`` for names missing from the
        local table. Off by default so the function is deterministic and
        network-free for testing.

    Returns
    -------
    pd.DataFrame
        One row per input name, indexed by the original (un-normalized)
        string, with one column per requested target.
    """
    lookup = _load_lookup()
    # Build a name → row index using canonical + aliases
    alias_to_row: dict[str, int] = {}
    for i, row in lookup.iterrows():
        alias_to_row[row["name"]] = i
        for a in (row.get("aliases") or "").split(";"):
            a = normalize_name(a)
            if a and a not in alias_to_row:
                alias_to_row[a] = i

    rows = []
    missing = []
    for n in names:
        key = normalize_name(n)
        idx = alias_to_row.get(key)
        if idx is None:
            rows.append({t: "" for t in targets})
            missing.append(n)
        else:
            rows.append({t: lookup.at[idx, t] for t in targets})

    out = pd.DataFrame(rows, index=list(names))
    if missing and allow_online:
        out = _online_fallback(out, missing, targets)
    return out


def _online_fallback(table: pd.DataFrame, missing: list[str],
                     targets: tuple[str, ...]) -> pd.DataFrame:
    """Best-effort lookup via bioservices (ChEBI / KEGG REST).

    Results cached in ``~/.cache/omicverse/metabol/online_cache.parquet``
    so repeated calls are free.
    """
    try:
        from bioservices import ChEBI, KEGG  # noqa: F401
    except ImportError:  # pragma: no cover
        # Silently skip — offline environments just won't resolve missing names.
        return table

    cache_dir = Path.home() / ".cache" / "omicverse" / "metabol"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "online_cache.parquet"
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()

    # Skip network calls that are already in cache
    new_rows = []
    for name in missing:
        key = normalize_name(name)
        hit = cache[cache["name"] == key] if "name" in cache.columns else None
        if hit is not None and len(hit) > 0:
            for t in targets:
                if t in hit.columns:
                    table.loc[name, t] = hit.iloc[0][t]
            continue
        # Online query — ChEBI first, then KEGG
        row = {"name": key}
        try:
            ch = ChEBI()
            res = ch.getLiteEntity(key, searchCategory="ALL NAMES")
            if isinstance(res, list) and res:
                row["chebi"] = res[0].get("chebiId", "")
        except Exception:  # pragma: no cover
            pass
        try:
            kg = KEGG()
            found = kg.find("compound", key)
            if found:
                row["kegg"] = found.split("\t")[0].replace("cpd:", "")
        except Exception:  # pragma: no cover
            pass
        for t in targets:
            if t in row and row[t]:
                table.loc[name, t] = row[t]
        new_rows.append(row)

    # Update cache
    if new_rows:
        new_cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        new_cache.drop_duplicates("name", keep="last").to_parquet(cache_path)
    return table


def build_lookup(out_path: Optional[Path] = None) -> Path:
    """Build the local metabolite lookup parquet from a curated CSV.

    This is what users would run once after a fresh install if the
    shipped parquet is missing or out of date. The curated CSV lives at
    ``omicverse/metabol/data/metabolite_lookup.csv``; we convert to
    parquet for fast load.
    """
    csv_path = _DATA_DIR / "metabolite_lookup.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Curated CSV missing at {csv_path}. "
            "The installation is incomplete; reinstall omicverse."
        )
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # Store monoisotopic mass as float64 (rest stay string for consistency)
    if "mw" in df.columns:
        df["mw"] = pd.to_numeric(df["mw"], errors="coerce")
    out_path = out_path or _LOOKUP_PATH
    df.to_parquet(out_path, index=False)
    return out_path


def available_metabolites() -> pd.DataFrame:
    """Return the full local lookup table (for discovery / debugging)."""
    return _load_lookup().copy()
