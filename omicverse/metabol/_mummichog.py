r"""Mummichog — pathway inference directly from m/z peaks.

For untargeted LC-MS data where you have m/z values but no compound
annotations, mummichog (Li et al 2013) does two things:

1. **Adduct-aware mass matching** — each m/z peak could represent
   several candidate KEGG compounds (e.g. ``[M+H]+``, ``[M+Na]+``,
   ``[M-H]-``). We enumerate adducts, compute theoretical m/z, and
   keep candidates within a ppm tolerance.
2. **Pathway enrichment by sampling** — the match ambiguity means a
   single "hit peak" votes for multiple candidate pathways. Mummichog
   samples null hit-lists from the input list and computes empirical
   p-values per pathway.

This module offers two entry points:

- :func:`mummichog_basic` — a pure-Python implementation that covers
  the common positive-mode adducts and the local KEGG pathway table.
  Fast, deterministic, no external tool required.
- :func:`mummichog_external` — thin wrapper around the ``mummichog``
  PyPI package (Li's reference implementation) for users who want the
  full set of adducts, activity-network scoring, and LIBSDB support.

Both take an input of peak m/z values + a ranking metric (p-value or
fold-change) and return a pathway enrichment table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ._id_mapping import _load_lookup
from ._msea import _bh_fdr, load_pathways


# Common ESI adducts — (name, delta_mass, charge_sign)
# delta is the mass shift from neutral monoisotopic mass M
POSITIVE_ADDUCTS = [
    ("M+H",       +1.00728, "+"),
    ("M+Na",      +22.98922, "+"),
    ("M+K",       +38.96316, "+"),
    ("M+NH4",     +18.03437, "+"),
    ("M+H-H2O",   +1.00728 - 18.01056, "+"),
    ("2M+H",      +1.00728, "+"),      # dimer — treat M as M/2 instead
]

NEGATIVE_ADDUCTS = [
    ("M-H",       -1.00728, "-"),
    ("M-H2O-H",   -1.00728 - 18.01056, "-"),
    ("M+Cl",      +34.96940, "-"),
    ("M+FA-H",    +44.99765, "-"),     # [M+HCOO]-, formate adduct
]


def annotate_peaks(
    mz: np.ndarray,
    *,
    polarity: str = "positive",
    ppm: float = 10.0,
    custom_adducts: Optional[list[tuple[str, float, str]]] = None,
) -> pd.DataFrame:
    """Map a list of m/z peaks to candidate KEGG compounds via adduct search.

    Parameters
    ----------
    mz
        1-D array of experimental m/z values.
    polarity
        ``"positive"`` or ``"negative"`` — picks the adduct list. For
        mixed modes, pass a merged list via ``custom_adducts``.
    ppm
        Mass-matching tolerance in parts per million. 5 ppm is typical
        for Orbitrap; 10–20 ppm for QTOF.
    custom_adducts
        Override the default adduct list. Each entry is
        ``(name, delta_mass, sign_str)``.

    Returns
    -------
    pd.DataFrame
        One row per (m/z, adduct, candidate KEGG) match. Columns:
        ``mz`` (input), ``adduct``, ``kegg``, ``name``, ``delta_ppm``.
        Multiple candidate matches per m/z are normal.
    """
    if polarity == "positive":
        adducts = custom_adducts or POSITIVE_ADDUCTS
    elif polarity == "negative":
        adducts = custom_adducts or NEGATIVE_ADDUCTS
    else:
        raise ValueError(f"polarity must be 'positive' or 'negative', got {polarity!r}")

    lookup = _load_lookup()
    lookup = lookup[lookup["mw"].notna() & (lookup["kegg"] != "")].copy()
    masses = lookup["mw"].to_numpy()

    rows = []
    mz = np.asarray(mz, dtype=np.float64)
    for ad_name, ad_delta, _ in adducts:
        # For [2M+H], "observed m/z" = 2*M + delta; for everything else it's M + delta
        factor = 2.0 if ad_name.startswith("2M") else 1.0
        theor_mz = factor * masses + ad_delta  # shape (n_compounds,)
        for i, peak in enumerate(mz):
            tol = peak * ppm / 1e6
            hits = np.where(np.abs(theor_mz - peak) <= tol)[0]
            for h in hits:
                rows.append({
                    "mz": peak,
                    "peak_idx": i,
                    "adduct": ad_name,
                    "kegg": lookup.iloc[h]["kegg"],
                    "name": lookup.iloc[h]["name"],
                    "mw": masses[h],
                    "theor_mz": theor_mz[h],
                    "delta_ppm": (peak - theor_mz[h]) / peak * 1e6,
                })
    return pd.DataFrame(rows)


def mummichog_basic(
    mz: np.ndarray,
    pvalue: np.ndarray,
    *,
    polarity: str = "positive",
    ppm: float = 10.0,
    significance_cutoff: float = 0.05,
    n_perm: int = 1000,
    min_overlap: int = 2,
    pathways: Optional[dict[str, list[str]]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Pure-Python mummichog — pathway enrichment from m/z peaks.

    Parameters
    ----------
    mz
        m/z value per peak.
    pvalue
        Per-peak p-value from the upstream univariate test. Peaks with
        ``pvalue < significance_cutoff`` are the "hit" set; the rest
        form the background.
    polarity
        ``"positive"`` or ``"negative"``.
    ppm
        Mass-matching tolerance in ppm. Default 10.
    significance_cutoff
        Threshold that splits ``mz`` into hits vs background.
    n_perm
        Random permutations for the empirical null p-value. 1000 is
        fine for tutorials; bump to ≥10000 for publication.
    min_overlap
        Skip pathways with fewer than this many overlapping compounds.

    Returns
    -------
    pd.DataFrame
        ``pathway``, ``overlap``, ``set_size``, ``pvalue`` (empirical),
        ``padj`` (BH), ``observed_score``.
    """
    if pathways is None:
        pathways = load_pathways()

    mz = np.asarray(mz, dtype=np.float64)
    pvalue = np.asarray(pvalue, dtype=np.float64)
    if mz.shape != pvalue.shape:
        raise ValueError(
            f"mz and pvalue must have the same shape; got {mz.shape} vs {pvalue.shape}"
        )
    hit_mask = pvalue < significance_cutoff
    if hit_mask.sum() < 3:
        raise ValueError(
            f"Too few significant peaks ({hit_mask.sum()}) at p<{significance_cutoff}."
        )

    ann = annotate_peaks(mz, polarity=polarity, ppm=ppm)
    if ann.empty:
        raise ValueError(
            "No m/z peak matched any KEGG compound — check polarity, ppm, or "
            "extend metabolite_lookup.csv."
        )

    # Map peak_idx → set of candidate KEGG compounds (one peak can vote for many)
    peak_to_kegg = (
        ann.groupby("peak_idx")["kegg"].apply(lambda s: set(s)).to_dict()
    )
    # Which peaks ended up with any annotation at all? (the "usable" universe)
    usable = np.array([i in peak_to_kegg for i in range(mz.size)])
    hit_peaks = np.where(hit_mask & usable)[0]
    bg_peaks = np.where(usable)[0]

    # Observed pathway hits: union of candidate KEGG compounds for hit peaks
    hit_kegg = set().union(*(peak_to_kegg[i] for i in hit_peaks))

    rng = np.random.default_rng(seed)
    rows = []
    for pw_name, pw_ids in pathways.items():
        pw_set = set(pw_ids)
        overlap = len(hit_kegg & pw_set)
        if overlap < min_overlap:
            continue
        # Permutation null: sample len(hit_peaks) from bg_peaks, count pathway hits
        null = np.zeros(n_perm, dtype=int)
        for k in range(n_perm):
            null_peaks = rng.choice(bg_peaks, size=len(hit_peaks), replace=False)
            null_kegg = set().union(*(peak_to_kegg[i] for i in null_peaks))
            null[k] = len(null_kegg & pw_set)
        pvalue_emp = float(((null >= overlap).sum() + 1) / (n_perm + 1))
        rows.append({
            "pathway": pw_name,
            "overlap": overlap,
            "set_size": len(pw_set),
            "pvalue": pvalue_emp,
            "observed_score": overlap / max(len(pw_set), 1),
            "hit_kegg": ";".join(sorted(hit_kegg & pw_set)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["padj"] = _bh_fdr(out["pvalue"].to_numpy())
    return out.sort_values("pvalue").reset_index(drop=True)


def mummichog_external(
    mz: np.ndarray,
    pvalue: np.ndarray,
    retention_time: Optional[np.ndarray] = None,
    *,
    mode: str = "pos",
    significance_cutoff: float = 0.05,
    outdir: Optional[str] = None,
    **kwargs,
):
    """Thin wrapper around the ``mummichog`` PyPI package.

    This is Li's reference implementation, more featureful than our
    basic port (full adduct table, activity-network scoring, LIBSDB).
    Install with ``pip install mummichog`` and ensure you have the
    pathway files bundled with that package.

    Parameters
    ----------
    mz, pvalue
        Same as :func:`mummichog_basic`.
    retention_time
        Optional retention-time column (seconds). Some mummichog modes
        use RT for ambiguity tie-breaking.
    mode
        ``"pos"`` or ``"neg"``.
    outdir
        Directory for mummichog's HTML/CSV reports. If ``None``, a
        temporary directory is used.
    **kwargs
        Forwarded to ``mummichog.Analyzer`` — see the upstream docs.
    """
    try:
        from mummichog.main import main as mumm_main
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mummichog_external requires the mummichog PyPI package — "
            "`pip install mummichog`."
        ) from exc
    import tempfile

    # mummichog expects a file on disk
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        cols = ["mz", "rtime", "p-value", "t-score"]
        f.write("\t".join(cols) + "\n")
        for i, (m, p) in enumerate(zip(mz, pvalue)):
            rt = retention_time[i] if retention_time is not None else 0.0
            f.write(f"{m}\t{rt}\t{p}\t0\n")
        tmp_path = f.name

    outdir = outdir or tempfile.mkdtemp(prefix="mummichog_")
    # Call mummichog's CLI-equivalent API
    args = [
        "-f", tmp_path,
        "-o", outdir,
        "-m", mode,
        "-c", str(significance_cutoff),
    ]
    for k, v in kwargs.items():
        args.extend([f"--{k}", str(v)])
    # mumm_main prints and writes files; we capture the summary CSV
    mumm_main(args)
    # mummichog writes several files; the pathway enrichment is typically
    # ``<outdir>/tables/mcg_pathwayanalysis_*.tsv``
    summary = None
    for p in Path(outdir).rglob("mcg_pathwayanalysis*.tsv"):
        summary = pd.read_csv(p, sep="\t")
        break
    return summary
