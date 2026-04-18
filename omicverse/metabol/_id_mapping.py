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


from functools import lru_cache

_DATA_DIR = Path(__file__).parent / "data"
_LOOKUP_CSV = _DATA_DIR / "metabolite_lookup.csv"
_LOOKUP_PARQUET = _DATA_DIR / "metabolite_lookup.parquet"
# Backwards-compat alias kept for tests that reference the old name
_LOOKUP_PATH = _LOOKUP_PARQUET


@lru_cache(maxsize=1)
def _load_lookup() -> pd.DataFrame:
    """Return the metabolite lookup table.

    Reads the curated CSV (the source of truth), caches the parsed
    DataFrame in-memory for the rest of the process, and lazily
    regenerates the on-disk parquet cache so subsequent installs
    don't re-parse the CSV every import. Safe to call repeatedly —
    ``lru_cache`` makes the Python side O(1).
    """
    if _LOOKUP_PARQUET.exists():
        try:
            return pd.read_parquet(_LOOKUP_PARQUET)
        except Exception:
            # Fall through to CSV path — parquet may be corrupt
            pass
    if not _LOOKUP_CSV.exists():
        raise FileNotFoundError(
            f"lookup CSV missing at {_LOOKUP_CSV}. "
            "The omicverse install is incomplete; reinstall."
        )
    df = pd.read_csv(_LOOKUP_CSV, dtype=str, keep_default_na=False)
    if "mw" in df.columns:
        df["mw"] = pd.to_numeric(df["mw"], errors="coerce")
    # Best-effort parquet cache write. If the install dir is read-only
    # (e.g. site-packages under an admin install) silently skip.
    try:
        df.to_parquet(_LOOKUP_PARQUET, index=False)
    except Exception:
        pass
    return df


@lru_cache(maxsize=1)
def _alias_to_row() -> dict[str, int]:
    """Build the name → row-index map once per process."""
    lu = _load_lookup()
    mapping: dict[str, int] = {}
    for i, row in lu.iterrows():
        mapping[row["name"]] = i
        for a in (row.get("aliases") or "").split(";"):
            a_norm = normalize_name(a)
            if a_norm and a_norm not in mapping:
                mapping[a_norm] = i
    return mapping


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
    alias_map = _alias_to_row()      # process-cached; O(1) per call now

    rows = []
    missing = []
    for n in names:
        key = normalize_name(n)
        idx = alias_map.get(key)
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
    """Rebuild the parquet cache from the curated CSV.

    ``_load_lookup()`` regenerates the cache automatically on first use,
    so users don't usually need to call this. Kept public for explicit
    cache refresh after editing ``metabolite_lookup.csv`` in a dev checkout.
    """
    # Force invalidate the in-memory caches too
    _load_lookup.cache_clear()
    _alias_to_row.cache_clear()
    out_path = out_path or _LOOKUP_PARQUET
    if out_path.exists():
        out_path.unlink()
    df = _load_lookup()   # recomputes + writes parquet
    return out_path


def available_metabolites() -> pd.DataFrame:
    """Return the full local lookup table (for discovery / debugging)."""
    return _load_lookup().copy()
