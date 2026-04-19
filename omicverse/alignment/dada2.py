"""DADA2 backend for 16S ASV inference.

Wraps the pure-Python re-implementation `pydada2`
(https://github.com/omicverse/py-dada2) — no R / rpy2 dependency.

Submodule-style layout (like ``ov.alignment.vsearch``): each function is a
thin wrapper around the corresponding pydada2 call, plus one orchestrator
:func:`dada2_pipeline` that produces a samples × ASVs :class:`anndata.AnnData`
with the same schema as :func:`omicverse.alignment.amplicon_16s_pipeline`.

Install pydada2: ``pip install pydada2``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError as exc:  # pragma: no cover
    raise ImportError("anndata is required for ov.alignment.dada2") from exc

from .._registry import register_function
from ._cli_utils import ensure_dir


def _require_pydada2():
    try:
        import pydada2
    except ImportError as exc:
        raise ImportError(
            "ov.alignment.dada2 requires pydada2 — install with "
            "`pip install pydada2`."
        ) from exc
    return pydada2


_SampleTuple = Tuple[str, str, Optional[str]]


# ---------------------------------------------------------------------------
# Thin wrappers
# ---------------------------------------------------------------------------


@register_function(
    aliases=["dada2.filter_and_trim", "dada2_filter"],
    category="alignment",
    description="DADA2 quality filtering and truncation (pure-Python pydada2).",
    examples=[
        "ov.alignment.dada2.filter_and_trim(samples, output_dir, trunc_len=(240,160), max_ee=(2,2))",
    ],
    related=["alignment.dada2", "alignment.dada2_pipeline"],
)
def filter_and_trim(
    samples: Union[_SampleTuple, Sequence[_SampleTuple]],
    output_dir: str,
    trunc_len: Union[int, Tuple[int, int]] = 0,
    max_ee: Union[float, Tuple[float, float]] = 2.0,
    trunc_q: int = 2,
    min_len: int = 20,
    max_n: int = 0,
    overwrite: bool = False,
) -> List[Dict[str, str]]:
    """Quality-trim each sample's FASTQs to `output_dir/<sample>/`.

    Accepts ``(sample, fq1, fq2)`` tuples; ``fq2`` may be None (single-end).
    """
    pydada2 = _require_pydada2()

    if isinstance(samples, tuple) and len(samples) == 3:
        sample_list: List[_SampleTuple] = [samples]
    else:
        sample_list = [tuple(s) for s in samples]  # type: ignore[arg-type]

    out_root = ensure_dir(output_dir)
    results: List[Dict[str, str]] = []

    # Normalise trunc_len / max_ee to PE tuple if needed downstream.
    def _norm_pair(x):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return tuple(x)
        return (x, x)

    for sample, fq1, fq2 in sample_list:
        sample_dir = ensure_dir(Path(out_root) / sample)
        src1 = Path(fq1)
        filt1 = sample_dir / f"{sample}_F_filt.fastq.gz"
        filt2 = sample_dir / f"{sample}_R_filt.fastq.gz" if fq2 else None

        if not overwrite and filt1.exists() and filt1.stat().st_size > 0:
            if not fq2 or (filt2 and filt2.exists() and filt2.stat().st_size > 0):
                results.append({
                    "sample": sample,
                    "filt1": str(filt1),
                    "filt2": str(filt2) if filt2 else "",
                })
                continue

        # pydada2 uses R DADA2's camelCase kwarg names
        if fq2:
            tl = _norm_pair(trunc_len) if trunc_len else 0
            me = _norm_pair(max_ee)
            pydada2.filter_and_trim(
                fwd=str(src1), filt=str(filt1),
                rev=str(fq2), filt_rev=str(filt2),
                truncLen=tl, maxEE=me, truncQ=trunc_q,
                minLen=min_len, maxN=max_n,
            )
        else:
            tl = trunc_len[0] if isinstance(trunc_len, (tuple, list)) else trunc_len
            me = max_ee[0] if isinstance(max_ee, (tuple, list)) else max_ee
            pydada2.filter_and_trim(
                fwd=str(src1), filt=str(filt1),
                truncLen=tl or 0, maxEE=me,
                truncQ=trunc_q, minLen=min_len, maxN=max_n,
            )

        if not filt1.exists() or filt1.stat().st_size == 0:
            raise RuntimeError(f"DADA2 filter_and_trim produced no output for {sample}")
        results.append({
            "sample": sample,
            "filt1": str(filt1),
            "filt2": str(filt2) if filt2 else "",
        })
    return results


@register_function(
    aliases=["dada2.learn_errors", "dada2_errors"],
    category="alignment",
    description="Learn DADA2 position × transition error rates from filtered reads (pure-Python pydada2).",
    examples=[
        "errF = ov.alignment.dada2.learn_errors([r['filt1'] for r in results])",
    ],
    related=["alignment.dada2"],
)
def learn_errors(
    fastqs: Union[str, Sequence[str]],
    nbases: int = 100_000_000,
    random_state: int = 0,
) -> "np.ndarray":
    """Learn a DADA2 error model from one or several filtered FASTQs.

    Returns the ``(nuc, nuc, qual)`` error-rate tensor that
    :func:`denoise` consumes.
    """
    pydada2 = _require_pydada2()
    return pydada2.learn_errors(
        fastqs, nbases=nbases, random_state=random_state,
    )


@register_function(
    aliases=["dada2.denoise", "dada2_dada"],
    category="alignment",
    description="Run the DADA divisive-amplicon-denoising algorithm on a filtered FASTQ (pydada2).",
    examples=[
        "dadaF = ov.alignment.dada2.denoise('filt/S1_F.fastq.gz', err=errF)",
    ],
    related=["alignment.dada2"],
)
def denoise(
    derep,
    err: Optional["np.ndarray"] = None,
    **kwargs,
):
    """Dispatch to ``pydada2.dada`` (the divisive denoiser)."""
    pydada2 = _require_pydada2()
    return pydada2.dada(derep, err=err, **kwargs)


@register_function(
    aliases=["dada2.merge_pairs", "dada2_merge"],
    category="alignment",
    description="Merge paired-end DADA2 denoised results (pydada2).",
    examples=[
        "mergers = ov.alignment.dada2.merge_pairs(ddF, filt1, ddR, filt2)",
    ],
    related=["alignment.dada2"],
)
def merge_pairs(dadaF, derepF, dadaR, derepR,
                min_overlap: int = 12, max_mismatch: int = 0):
    pydada2 = _require_pydada2()
    return pydada2.merge_pairs(
        dadaF, derepF, dadaR, derepR,
        minOverlap=min_overlap, maxMismatch=max_mismatch,
    )


@register_function(
    aliases=["dada2.make_seqtab", "dada2_seqtab"],
    category="alignment",
    description="Build a sample × ASV-sequence abundance table from DADA2 mergers.",
    examples=[
        "seqtab = ov.alignment.dada2.make_seqtab(mergers_by_sample)",
    ],
    related=["alignment.dada2"],
)
def make_seqtab(samples_mergers, order_by: str = "abundance") -> pd.DataFrame:
    pydada2 = _require_pydada2()
    return pydada2.make_sequence_table(samples_mergers, orderBy=order_by)


@register_function(
    aliases=["dada2.remove_chimeras", "dada2_chimera"],
    category="alignment",
    description="De novo chimera removal on a DADA2 sequence table (consensus method).",
    examples=[
        "seqtab_nochim = ov.alignment.dada2.remove_chimeras(seqtab)",
    ],
    related=["alignment.dada2"],
)
def remove_chimeras(seqtab, method: str = "consensus", verbose: bool = False):
    pydada2 = _require_pydada2()
    return pydada2.remove_bimera_denovo(seqtab, method=method, verbose=verbose)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _seqtab_to_anndata(
    seqtab: pd.DataFrame,
    sample_order: Sequence[str],
) -> "ad.AnnData":
    """Convert DADA2 sample × sequence table to AnnData with ASVn ids."""
    from scipy import sparse
    # pydada2 returns samples as the index, sequences as columns
    seqtab = seqtab.reindex(sample_order)
    sequences = list(seqtab.columns)
    asv_ids = [f"ASV{i+1}" for i in range(len(sequences))]

    X = sparse.csr_matrix(
        np.nan_to_num(seqtab.values, nan=0.0).astype(np.int32)
    )
    obs = pd.DataFrame(
        {"sample": list(sample_order)},
        index=pd.Index(list(sample_order), name="sample"),
    )
    var = pd.DataFrame(
        {"sequence": sequences},
        index=pd.Index(asv_ids, name="asv"),
    )
    # populate empty taxonomy columns (aligns with SINTAX schema)
    for col in ("domain", "phylum", "class", "order", "family",
                "genus", "species", "taxonomy"):
        var[col] = ""
    var["sintax_confidence"] = np.nan
    var["sintax_raw"] = ""
    return ad.AnnData(X=X, obs=obs, var=var)


@register_function(
    aliases=[
        "dada2_pipeline", "amplicon_16s_dada2_pipeline",
        "dada2.pipeline",
    ],
    category="alignment",
    description="End-to-end pydada2 pipeline: filter → learn_errors → denoise → merge_pairs → seqtab → remove_chimeras → AnnData.",
    examples=[
        "adata = ov.alignment.dada2_pipeline(samples, workdir='/scratch/.../run')",
    ],
    related=[
        "alignment.amplicon_16s_pipeline", "alignment.dada2",
        "alignment.vsearch",
    ],
)
def dada2_pipeline(
    samples: Sequence[_SampleTuple],
    workdir: Optional[str] = None,
    *,
    db_fasta: Optional[str] = None,
    trunc_len: Union[int, Tuple[int, int]] = (240, 160),
    max_ee: Union[float, Tuple[float, float]] = (2.0, 2.0),
    trunc_q: int = 2,
    min_overlap: int = 12,
    max_mismatch: int = 0,
    chimera_method: str = "consensus",
    nbases: int = 100_000_000,
    sintax_cutoff: float = 0.8,
    sintax_strand: str = "both",
    threads: int = 4,
    sample_metadata: Optional[pd.DataFrame] = None,
    overwrite: bool = False,
):
    """Run pydada2 end-to-end and return an AnnData.

    Parameters
    ----------
    samples
        ``[(sample, fq1, fq2), ...]``. ``fq2`` may be None for single-end.
    workdir
        Required. Absolute path for intermediates. No ``$HOME`` fallback.
    db_fasta
        Optional path to a SINTAX-formatted reference FASTA. When supplied,
        ASV taxonomy is assigned by piping the ASV FASTA through
        :func:`omicverse.alignment.vsearch.sintax` and merging the 7-rank
        call into ``adata.var``.
    trunc_len
        ``(fwd, rev)`` truncation lengths in bp. V4 default ``(240, 160)``.
    max_ee
        ``(fwd, rev)`` expected-error thresholds.
    chimera_method
        Passed to ``pydada2.remove_bimera_denovo``.
    threads
        Used only by the vsearch SINTAX pass at the end.
    """
    if not workdir:
        raise ValueError(
            "`workdir` is required. omicverse never writes DADA2 "
            "intermediates to an implicit location."
        )
    if not samples:
        raise ValueError("`samples` is empty.")
    pydada2 = _require_pydada2()

    workdir = str(Path(workdir).expanduser().resolve())
    ensure_dir(workdir)

    sample_list: List[_SampleTuple] = [tuple(s) for s in samples]
    sample_order = [s[0] for s in sample_list]
    has_rev = any(s[2] for s in sample_list)

    paths: Dict[str, str] = {"workdir": workdir}

    # --- 1. Filter + trim ----------------------------------------------
    filt_dir = str(Path(workdir) / "filtered")
    filt_results = filter_and_trim(
        sample_list,
        output_dir=filt_dir,
        trunc_len=trunc_len,
        max_ee=max_ee,
        trunc_q=trunc_q,
        overwrite=overwrite,
    )
    paths["filtered"] = filt_dir

    # --- 2. Learn error rates on combined F / R sets -------------------
    # pydada2.learn_errors returns {'err_out': np.ndarray, 'err_in', 'trans'};
    # pydada2.dada wants the error matrix directly (the 'err_out' array).
    filt_f = [r["filt1"] for r in filt_results]
    errF_res = pydada2.learn_errors(filt_f, nbases=nbases)
    errF = errF_res["err_out"] if isinstance(errF_res, dict) else errF_res
    errR = None
    if has_rev:
        filt_r = [r["filt2"] for r in filt_results if r["filt2"]]
        errR_res = pydada2.learn_errors(filt_r, nbases=nbases)
        errR = errR_res["err_out"] if isinstance(errR_res, dict) else errR_res

    # --- 3. Denoise + merge per sample ---------------------------------
    samples_mergers: Dict[str, object] = {}
    for r in filt_results:
        sample = r["sample"]
        derepF = pydada2.derep_fastq(r["filt1"])
        ddF = pydada2.dada(derepF, err=errF)
        if has_rev and r["filt2"]:
            derepR = pydada2.derep_fastq(r["filt2"])
            ddR = pydada2.dada(derepR, err=errR)
            merged = pydada2.merge_pairs(
                ddF, derepF, ddR, derepR,
                minOverlap=min_overlap, maxMismatch=max_mismatch,
            )
            samples_mergers[sample] = merged
        else:
            # single-end: use the ddF result directly (pydada2 accepts it in make_sequence_table)
            samples_mergers[sample] = ddF

    # --- 4. Sequence table + chimera removal ---------------------------
    seqtab = pydada2.make_sequence_table(samples_mergers)
    seqtab_nochim = pydada2.remove_bimera_denovo(seqtab, method=chimera_method)
    paths["seqtab_before_chim"] = str(Path(workdir) / "seqtab.parquet")
    paths["seqtab_nochim"] = str(Path(workdir) / "seqtab_nochim.parquet")
    seqtab.to_parquet(paths["seqtab_before_chim"])
    seqtab_nochim.to_parquet(paths["seqtab_nochim"])

    # --- 5. AnnData assembly -------------------------------------------
    adata = _seqtab_to_anndata(seqtab_nochim, sample_order=sample_order)

    # Write ASV FASTA (ASVn → sequence) so SINTAX can consume it
    asv_fasta = Path(workdir) / "asv" / "asvs.fasta"
    asv_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(asv_fasta, "w", encoding="utf-8") as fh:
        for asv_id, seq in zip(adata.var_names, adata.var["sequence"]):
            fh.write(f">{asv_id}\n{seq}\n")
    paths["asv_fasta"] = str(asv_fasta)

    # --- 6. Optional taxonomy via vsearch SINTAX ------------------------
    if db_fasta:
        from . import vsearch as _vsearch_mod
        from .amplicon_16s import _parse_sintax_tsv
        tax_dir = str(Path(workdir) / "taxonomy")
        sintax_out = _vsearch_mod.sintax(
            str(asv_fasta),
            db_fasta=db_fasta,
            output_dir=tax_dir,
            cutoff=sintax_cutoff,
            strand=sintax_strand,
            threads=threads,
            overwrite=overwrite,
        )
        tax_df = _parse_sintax_tsv(sintax_out["tsv"])
        tax_df = tax_df.reindex(adata.var_names)
        for col in ("domain", "phylum", "class", "order", "family",
                    "genus", "species", "taxonomy", "sintax_raw"):
            if col in tax_df.columns:
                adata.var[col] = tax_df[col].fillna("").values
        if "sintax_confidence" in tax_df.columns:
            adata.var["sintax_confidence"] = tax_df["sintax_confidence"].values
        paths["taxonomy_tsv"] = sintax_out["tsv"]

    # --- 7. Merge user metadata ----------------------------------------
    if sample_metadata is not None:
        meta = sample_metadata.copy()
        meta.index = meta.index.astype(str)
        adata.obs = adata.obs.join(meta, how="left")

    adata.uns["pipeline"] = {
        "tool": "pydada2 (pure-Python DADA2)",
        "backend": "dada2",
        "paths": paths,
        "params": {
            "trunc_len": trunc_len,
            "max_ee": max_ee,
            "trunc_q": trunc_q,
            "min_overlap": min_overlap,
            "max_mismatch": max_mismatch,
            "chimera_method": chimera_method,
            "db_fasta": db_fasta,
            "sintax_cutoff": sintax_cutoff,
            "sintax_strand": sintax_strand,
            "threads": threads,
        },
    }
    return adata
