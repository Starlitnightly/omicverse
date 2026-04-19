r"""Cross-reference metabolite names ↔ HMDB / KEGG / ChEBI / LipidMaps IDs.

The lookup is **online-first**, cached forever:

1. :func:`map_ids` calls :func:`omicverse.metabol.fetch_hmdb_from_name`
   per name, which hits the PubChem REST API (synonym-aware) and pulls
   HMDB / KEGG / ChEBI IDs from the cross-reference list in one call.
2. Every resolved name is cached at
   ``~/.cache/omicverse/metabol/pubchem_xref_cache.json`` so repeat
   calls for the same compound name are free.
3. For **bulk offline** operation (e.g. air-gapped CI), call
   :func:`omicverse.metabol.fetch_chebi_compounds` once — it downloads
   ~15 MB from EBI and gives you a ``DataFrame`` of ~54k compounds
   with cross-references, which you can then filter offline.

There is no shipped name→ID lookup anymore. The previous curated
``metabolite_lookup.csv`` was removed because (a) its 95 entries were
dwarfed by the tutorials' real needs and (b) hardcoding a small subset
was misleading users into thinking the package had limited coverage.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .._registry import register_function


def normalize_name(s: str) -> str:
    """Lowercase + collapse whitespace — the canonical form for caching."""
    return " ".join(str(s).strip().lower().split())


@register_function(
    aliases=[
        'map_ids',
        '代谢物ID映射',
        'hmdb_kegg_chebi',
    ],
    category='metabolomics',
    description='Resolve metabolite names to HMDB / KEGG / ChEBI / PubChem IDs via PubChem REST (cached). Pass mass_db=fetch_chebi_compounds() to avoid per-name HTTP round-trips.',
    examples=[
        "ov.metabol.map_ids(['Glucose', 'Isoleucine'])",
    ],
    related=[
        'metabol.fetch_hmdb_from_name',
        'metabol.fetch_chebi_compounds',
    ],
)
def map_ids(
    names: Iterable[str],
    *,
    targets: tuple[str, ...] = ("hmdb", "kegg", "chebi"),
    mass_db: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve metabolite names to external database IDs.

    Parameters
    ----------
    names
        Iterable of metabolite names (e.g. ``adata.var_names``).
    targets
        Which external IDs to resolve — any subset of
        ``("hmdb", "kegg", "chebi", "pubchem", "lipidmaps")``.
    mass_db
        Optional pre-fetched ChEBI DataFrame from
        :func:`fetch_chebi_compounds`. When supplied, we look the name
        up in ``mass_db["name"]`` **first** (instant) and fall back to
        PubChem only for unresolved names. Recommended for workflows
        that call ``map_ids`` many times in a loop: fetch the DB once
        and pass it every call to avoid per-name HTTP round-trips.

    Returns
    -------
    pd.DataFrame
        One row per input name, indexed by the original (un-normalized)
        string, with one column per requested target. Empty string for
        unresolved targets.
    """
    from ._fetchers import fetch_hmdb_from_name

    targets = tuple(targets)
    rows: list[dict[str, str]] = []
    idx_of_name: dict[str, int] | None = None
    if mass_db is not None and "name" in mass_db.columns:
        idx_of_name = {
            normalize_name(n): i for i, n in enumerate(mass_db["name"])
        }

    for name in names:
        row = {t: "" for t in targets}
        if idx_of_name is not None:
            hit = idx_of_name.get(normalize_name(name))
            if hit is not None:
                for t in targets:
                    if t in mass_db.columns:
                        v = mass_db.iloc[hit][t]
                        if isinstance(v, str) and v:
                            row[t] = v
        # Fall back to PubChem per-name for anything still empty
        if any(not row[t] for t in targets):
            try:
                ids = fetch_hmdb_from_name(name)
            except Exception:
                ids = {}
            for t in targets:
                if not row[t] and ids.get(t):
                    row[t] = ids[t]
        rows.append(row)

    return pd.DataFrame(rows, index=list(names))
