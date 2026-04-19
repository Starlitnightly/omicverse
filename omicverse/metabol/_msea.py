r"""Metabolite-Set Enrichment Analysis (MSEA).

Two enrichment flavours, both keyed off KEGG-compound pathway membership:

1. **ORA** (over-representation analysis) — classic Fisher's-exact on
   a pre-selected hit list vs the background universe. Fast, no
   ranking needed.
2. **GSEA-style** — full ranked-list enrichment via ``gseapy.prerank``.
   Uses the Welch-t statistic (or any user-supplied metric) as the
   ranking; output schema matches ``gseapy.GSEA``.

Both use the local pathway table at
``omicverse/metabol/data/kegg_pathways.csv`` by default, and translate
metabolite names → KEGG compound IDs via :mod:`omicverse.metabol._id_mapping`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ._id_mapping import map_ids, normalize_name
from ._utils import bh_fdr as _bh_fdr


def load_pathways(
    path: Optional[Path] = None,
    *,
    organism: Optional[str] = None,
) -> dict[str, list[str]]:
    """Return ``{pathway_name: [kegg_id, ...]}`` — the pathway database
    used by :func:`msea_ora` / :func:`msea_gsea` / :func:`mummichog_basic`.

    The default is the **full KEGG pathway database** fetched from the
    public KEGG REST endpoint (cached under
    ``~/.cache/omicverse/metabol/``; ~550 pathways). First call needs
    network; subsequent calls are free.

    Parameters
    ----------
    path
        Override with your own pathway CSV (columns: ``pathway_name``,
        ``kegg_compounds`` — the latter a ``;``-joined list of KEGG IDs).
        Useful for domain-specific pathway collections (e.g. SMPDB,
        Reactome-metabolite, or a curated clinical panel). Skips the
        network entirely.
    organism
        KEGG organism code for species-specific pathways (``"hsa"``
        human, ``"mmu"`` mouse, …). Default ``None`` → species-agnostic
        ``map#####`` reference metabolic pathways, which is what
        enrichment papers usually use.
    """
    if path is not None:
        df = pd.read_csv(path)
        return {
            row["pathway_name"]: row["kegg_compounds"].split(";")
            for _, row in df.iterrows()
        }
    from ._fetchers import fetch_kegg_pathways
    return fetch_kegg_pathways(organism=organism)


def msea_ora(
    hits: Iterable[str],
    background: Iterable[str],
    *,
    pathways: Optional[dict[str, list[str]]] = None,
    min_size: int = 3,
) -> pd.DataFrame:
    """Over-representation analysis via Fisher's exact test.

    Parameters
    ----------
    hits
        Metabolite names (e.g. from ``pyMetabo.significant_metabolites()``).
    background
        All tested metabolite names (the universe). Usually
        ``adata.var_names`` after filtering.
    pathways
        Optional override of ``{pathway_name: [kegg_id, ...]}``. Default
        is the local KEGG subset shipped with omicverse.
    min_size
        Skip pathways with fewer than this many overlapping background
        compounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``pathway``, ``overlap``, ``set_size``, ``universe_size``,
        ``odds_ratio``, ``pvalue``, ``padj`` (BH).
    """
    if pathways is None:
        pathways = load_pathways()
    # Map names → KEGG IDs
    hit_kegg = set(map_ids(list(hits))["kegg"].dropna().tolist()) - {""}
    bg_kegg = set(map_ids(list(background))["kegg"].dropna().tolist()) - {""}
    if not hit_kegg:
        raise ValueError(
            "None of the hit metabolite names resolve to KEGG compound IDs — "
            "check spelling or extend metabolite_lookup.csv."
        )

    rows = []
    for pw_name, pw_ids in pathways.items():
        pw_set = set(pw_ids) & bg_kegg
        if len(pw_set) < min_size:
            continue
        # 2x2 contingency: in_hit & in_pw | in_hit & not_pw
        #                  not_hit & in_pw | not_hit & not_pw
        overlap = hit_kegg & pw_set
        a = len(overlap)
        b = len(hit_kegg - pw_set)
        c = len(pw_set - hit_kegg)
        d = len(bg_kegg - hit_kegg - pw_set)
        if a == 0:
            continue
        try:
            odds_ratio, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        except ValueError:
            continue
        rows.append({
            "pathway": pw_name,
            "overlap": a,
            "set_size": len(pw_set),
            "universe_size": len(bg_kegg),
            "odds_ratio": odds_ratio,
            "pvalue": pvalue,
            "hit_kegg": ";".join(sorted(overlap)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["padj"] = _bh_fdr(out["pvalue"].to_numpy())
    return out.sort_values("pvalue").reset_index(drop=True)


def msea_gsea(
    deg: pd.DataFrame,
    *,
    stat_col: str = "stat",
    pathways: Optional[dict[str, list[str]]] = None,
    n_perm: int = 1000,
    min_size: int = 3,
    max_size: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """GSEA-style ranked enrichment via ``gseapy.prerank``.

    Parameters
    ----------
    deg
        Output DataFrame from :func:`differential`. Rows indexed by
        metabolite name; column ``stat_col`` provides the ranking metric.
    stat_col
        Which column of ``deg`` to rank on. Default ``"stat"`` (signed
        t-statistic); ``"log2fc"`` is another common choice.
    pathways
        Dict mapping pathway name to list of KEGG compound IDs.
    n_perm
        Permutation count for the empirical null. 1000 is fine for
        tutorials; bump to ≥10000 for publication.

    Returns
    -------
    pd.DataFrame
        Columns: ``Term``, ``NES``, ``NOM p-val``, ``FDR q-val``,
        ``ES``, ``Lead_genes`` (metabolites driving the enrichment).
    """
    # Use the vendored gseapy that ships inside omicverse.external so we
    # don't pin a separate top-level dependency (avoids version conflicts
    # with other GSEA-using modules).
    try:
        from ..external.gseapy import prerank as _prerank
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "msea_gsea requires omicverse.external.gseapy — this should "
            "be bundled with omicverse; reinstall if missing."
        ) from exc

    if pathways is None:
        pathways = load_pathways()
    # Build rank: metabolite-name → score
    rank_df = deg[[stat_col]].copy()
    rank_df["name"] = rank_df.index
    rank_df = rank_df.reset_index(drop=True)
    # Resolve names → KEGG
    id_map = map_ids(rank_df["name"].tolist())
    rank_df["kegg"] = id_map["kegg"].values
    rank_df = rank_df[rank_df["kegg"] != ""].drop_duplicates("kegg")
    if rank_df.empty:
        raise ValueError(
            "None of the differential-result metabolites resolve to KEGG IDs."
        )
    rnk = rank_df.set_index("kegg")[stat_col].sort_values(ascending=False)

    result = _prerank(
        rnk=rnk, gene_sets=pathways, min_size=min_size, max_size=max_size,
        permutation_num=n_perm, outdir=None, seed=seed, no_plot=True, verbose=False,
    )
    out = result.res2d.reset_index(drop=True) if hasattr(result, "res2d") else pd.DataFrame()
    return out


