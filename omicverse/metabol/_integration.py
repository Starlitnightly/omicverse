r"""Multi-omics integration for metabolomics.

A metabolomics study is rarely standalone — the clinical or biological
signal it carries is usually clearer when joined with RNA-seq,
proteomics, or lipidomics from the **same samples**. The standard
unsupervised factor model for this is MOFA+ (Argelaguet 2020), which
is already vendored in ``omicverse.external.mofapy2`` and wrapped at
``ov.single.pyMOFA``.

The wrapper at ``single.pyMOFA`` is a thin adapter that feeds an
``omics`` list into ``mofapy2.entry_point``, and its *training path*
is omics-agnostic — the MOFA(+) engine works for bulk or single-cell
alike. But its *downstream helpers* (``pyMOFAART``,
``factor_exact``, ``factor_correlation``) assume a single-cell
AnnData with ``obs`` rows that are *cells*. That's the wrong mental
model for metabolomics, where each ``obs`` row is a sample.

:func:`run_mofa` is a metabolomics-friendly bridge: it takes a dict
of aligned-sample AnnData views, validates the alignment, drives
``pyMOFA.mofa_run``, parses the resulting HDF5, and returns the
per-sample factor matrix ready to attach to ``adata.obsm['X_mofa']``
for any downstream omicverse step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function


@register_function(
    aliases=[
        'run_mofa',
        'mofa_metabol',
        'multi_omics_metabol',
        '多组学整合',
    ],
    category='metabolomics',
    description='Metabolomics-friendly bridge around ov.single.pyMOFA / mofapy2 — takes a dict of sample-aligned AnnData views and returns a per-sample factor matrix. Avoids the scRNA-seq-oriented downstream helpers.',
    examples=[
        "ov.metabol.run_mofa({'metabol': adata_m, 'rna': adata_r}, n_factors=10)",
    ],
    related=[
        'metabol.asca',
        'metabol.plsda',
    ],
)
def run_mofa(
    views: dict,
    *,
    n_factors: int = 10,
    outfile: str | Path = "mofa_model.hdf5",
    scale_views: bool = True,
    center_groups: bool = True,
    max_iter: int = 500,
    convergence_mode: str = "fast",
    gpu_mode: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Train MOFA+ on sample-aligned metabolomics + other-omics views.

    Parameters
    ----------
    views
        Mapping ``{view_name: AnnData}``. Each AnnData must have the
        **same ``obs_names`` in the same order** — MOFA+ concatenates
        samples across views by position. Mismatched indices raise a
        ``ValueError`` with the first few offending sample IDs.
    n_factors
        Number of latent factors. Default 10 — MOFA+ drops factors
        with variance explained below ``dropR2=0.001`` automatically.
    outfile
        Path to the HDF5 the trainer writes. Kept on disk so the user
        can reload it with ``ov.single.pyMOFAART`` if they want to
        drive the sc-oriented downstream helpers on their own.
    scale_views
        Scale each view to unit total variance before training. The
        MOFA+ recommendation when view dimensions / dynamic ranges
        differ by >10x (metabolomics vs RNA-seq definitely qualifies).
    center_groups
        Mean-centre each (view, sample-group) block before training.
    max_iter, convergence_mode
        Passed to ``pyMOFA.mofa_run``.
    gpu_mode
        Pass through to MOFA's GPU path. Requires CuPy.
    seed
        Deterministic seed.

    Returns
    -------
    pd.DataFrame
        ``(n_samples, n_factors_retained)`` factor matrix indexed by
        the shared sample IDs, columns ``F1, F2, ...``. To plug it
        into a downstream step:

        >>> factors = ov.metabol.run_mofa({'metabol': adata, 'rna': rna})
        >>> adata.obsm['X_mofa'] = factors.reindex(adata.obs_names).to_numpy()
    """
    if len(views) < 2:
        raise ValueError(
            f"run_mofa expects >=2 views (got {len(views)}). "
            f"For a single view use ov.metabol.plsda or sklearn PCA."
        )

    view_names = list(views.keys())
    adatas = [views[v] for v in view_names]

    # Sample-alignment check
    ref_names = list(adatas[0].obs_names)
    for name, ad in zip(view_names, adatas):
        cur = list(ad.obs_names)
        if cur != ref_names:
            # Find first mismatch for an informative error
            miss = []
            for i, (r, c) in enumerate(zip(ref_names, cur)):
                if r != c:
                    miss.append(f"pos {i}: {r!r} vs {c!r}")
                    if len(miss) >= 3:
                        break
            raise ValueError(
                f"view {name!r} obs_names do not match view "
                f"{view_names[0]!r}. First differences: {miss}. "
                f"Both views must list the same samples in the same order."
            )

    # Feed into pyMOFA
    try:
        from ..single._mofa import pyMOFA
    except Exception as exc:
        raise ImportError(
            "run_mofa needs omicverse.single.pyMOFA (vendored mofapy2). "
            "`pip install mofapy2 mofax` if the optional import fails."
        ) from exc

    mofa = pyMOFA(omics=adatas, omics_name=view_names)
    mofa.mofa_preprocess()
    out_path = str(Path(outfile).expanduser().resolve())
    mofa.mofa_run(
        outfile=out_path,
        factors=n_factors,
        iter=max_iter,
        convergence_mode=convergence_mode,
        gpu_mode=gpu_mode,
        seed=seed,
        scale_views=scale_views,
        center_groups=center_groups,
        verbose=False,
    )

    # Parse HDF5 — extract Z (factor × sample) and sample names
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("run_mofa needs h5py to parse the MOFA HDF5.") from exc

    with h5py.File(out_path, "r") as f:
        groups = list(f["groups"]["groups"].asstr())
        g0 = groups[0]
        Z = np.asarray(f["expectations"]["Z"][g0][:])  # (factors, samples)
        samples = list(f["samples"][g0].asstr())

    factors_df = pd.DataFrame(
        Z.T,
        index=samples,
        columns=[f"F{i + 1}" for i in range(Z.shape[0])],
    )
    factors_df.attrs["view_names"] = view_names
    factors_df.attrs["n_factors"] = int(Z.shape[0])
    factors_df.attrs["hdf5_path"] = out_path
    return factors_df
