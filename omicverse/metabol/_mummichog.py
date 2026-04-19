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

from ._fetchers import fetch_chebi_compounds
from ._msea import load_pathways
from ._utils import bh_fdr as _bh_fdr

from .._registry import register_function


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


@register_function(
    aliases=[
        'annotate_peaks',
        'adduct注释',
        'mz_to_kegg',
    ],
    category='metabolomics',
    description='Map m/z peaks to candidate KEGG compounds via adduct-aware mass matching. Default mass_db=fetch_chebi_compounds() (~54k compounds).',
    examples=[
        "ov.metabol.annotate_peaks(mz_values, polarity='positive', ppm=10.0)",
    ],
    related=[
        'metabol.mummichog_basic',
        'metabol.fetch_chebi_compounds',
    ],
)
def annotate_peaks(
    mz: np.ndarray,
    *,
    polarity: str = "positive",
    ppm: float = 10.0,
    custom_adducts: Optional[list[tuple[str, float, str]]] = None,
    mass_db: Optional[pd.DataFrame] = None,
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
    mass_db
        Compound master table with at least ``mw``, ``kegg``, ``name``
        columns. Defaults to ``ov.metabol.fetch_chebi_compounds()`` —
        ~54k ChEBI 3-star compounds with monoisotopic mass and
        KEGG/HMDB/LIPID MAPS cross-refs. Pass your own DataFrame to
        restrict the search space (e.g. only lipids, or a curated
        clinical panel).

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

    lookup = mass_db if mass_db is not None else fetch_chebi_compounds()
    for col in ("mw", "kegg", "name"):
        if col not in lookup.columns:
            raise ValueError(
                f"mass_db must have column {col!r}; got {list(lookup.columns)}"
            )
    lookup = lookup[lookup["mw"].notna() & (lookup["kegg"].astype(str) != "")].copy()
    masses = lookup["mw"].to_numpy(dtype=np.float64)
    kegg_arr = lookup["kegg"].to_numpy()
    name_arr = lookup["name"].to_numpy()

    mz = np.asarray(mz, dtype=np.float64)
    tol_per_peak = mz * ppm / 1e6       # ppm tolerance per peak, shape (n_peaks,)

    # Vectorized matching — broadcast (n_peaks, n_compounds) per adduct.
    # Still an O(n_adducts × n_peaks × n_compounds) arithmetic cost, but
    # the heavy lifting is one numpy call per adduct instead of a Python
    # triple-for. 30–100× speedup on real LC-MS inputs.
    frames = []
    for ad_name, ad_delta, _ in adducts:
        factor = 2.0 if ad_name.startswith("2M") else 1.0
        theor_mz = factor * masses + ad_delta                       # (n_compounds,)
        delta = mz[:, None] - theor_mz[None, :]                     # (n_peaks, n_compounds)
        match = np.abs(delta) <= tol_per_peak[:, None]
        peak_idx, cmpd_idx = np.where(match)
        if peak_idx.size == 0:
            continue
        frames.append(pd.DataFrame({
            "mz": mz[peak_idx],
            "peak_idx": peak_idx,
            "adduct": ad_name,
            "kegg": kegg_arr[cmpd_idx],
            "name": name_arr[cmpd_idx],
            "mw": masses[cmpd_idx],
            "theor_mz": theor_mz[cmpd_idx],
            "delta_ppm": delta[peak_idx, cmpd_idx] / mz[peak_idx] * 1e6,
        }))
    if not frames:
        return pd.DataFrame(columns=["mz", "peak_idx", "adduct", "kegg",
                                     "name", "mw", "theor_mz", "delta_ppm"])
    return pd.concat(frames, ignore_index=True)


@register_function(
    aliases=[
        'mummichog_basic',
        'mummichog',
        'm/z通路',
    ],
    category='metabolomics',
    description='Mummichog (Li et al 2013) — pathway enrichment from m/z peaks via adduct-aware mass matching + permutation null. Pure-Python port.',
    examples=[
        "ov.metabol.mummichog_basic(mz, pvalue, polarity='positive', ppm=10.0, n_perm=1000)",
    ],
    related=[
        'metabol.annotate_peaks',
        'metabol.mummichog_external',
    ],
)
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
    mass_db: Optional[pd.DataFrame] = None,
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

    ann = annotate_peaks(mz, polarity=polarity, ppm=ppm, mass_db=mass_db)
    if ann.empty:
        raise ValueError(
            "No m/z peak matched any KEGG compound at the requested polarity "
            "and ppm. Check that polarity matches the ionization mode of your "
            "experiment, loosen ppm (e.g. 20 for QTOF), or pass a smaller "
            "mass_db restricted to the compound classes you expect."
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
    # Pre-filter pathways to only those that are testable, so tqdm's total is accurate
    candidate_pathways = [
        (pw_name, set(pw_ids), len(hit_kegg & set(pw_ids)))
        for pw_name, pw_ids in pathways.items()
    ]
    candidate_pathways = [
        (n, s, o) for (n, s, o) in candidate_pathways if o >= min_overlap
    ]

    # Optional tqdm — long perm loops on many pathways benefit from a bar.
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(candidate_pathways, desc="mummichog", unit="pathway")
    except ImportError:  # pragma: no cover
        iterator = candidate_pathways

    for pw_name, pw_set, overlap in iterator:
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

    # mummichog expects a tab-separated file on disk. Use try/finally so
    # the temp file is unlinked even if mummichog crashes mid-run.
    import os as _os

    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    tmp_path = tmp.name
    try:
        cols = ["mz", "rtime", "p-value", "t-score"]
        tmp.write("\t".join(cols) + "\n")
        for i, (m, p) in enumerate(zip(mz, pvalue)):
            rt = retention_time[i] if retention_time is not None else 0.0
            tmp.write(f"{m}\t{rt}\t{p}\t0\n")
        tmp.close()

        outdir = outdir or tempfile.mkdtemp(prefix="mummichog_")
        args = [
            "-f", tmp_path,
            "-o", outdir,
            "-m", mode,
            "-c", str(significance_cutoff),
        ]
        for k, v in kwargs.items():
            args.extend([f"--{k}", str(v)])
        mumm_main(args)
        summary = None
        for p in Path(outdir).rglob("mcg_pathwayanalysis*.tsv"):
            summary = pd.read_csv(p, sep="\t")
            break
        return summary
    finally:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
