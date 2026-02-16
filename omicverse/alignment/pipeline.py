"""End-to-end bulk RNA-seq pipeline: SRA download -> QC -> alignment -> quantification."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .._registry import register_function
from .prefetch import prefetch
from .fq_dump import fqdump
from .fastp import fastp
from .STAR import STAR
from .featureCount import featureCount


@register_function(
    aliases=[
        "bulk_rnaseq_pipeline", "bulk_pipeline", "rnaseq_pipeline",
        "bulk_rnaseq", "RNA-seq流程", "bulk流程",
    ],
    category="alignment",
    description=(
        "End-to-end bulk RNA-seq pipeline: "
        "SRA download -> FASTQ QC -> STAR alignment -> featureCounts quantification."
    ),
    examples=[
        "ov.alignment.bulk_rnaseq_pipeline(['SRR1','SRR2'], genome_dir='index', gtf='genes.gtf')",
        "ov.alignment.bulk_rnaseq_pipeline(samples=[('S1','s1_1.fq.gz','s1_2.fq.gz')], genome_dir='idx', gtf='g.gtf', skip_download=True)",
    ],
    related=[
        "alignment.prefetch", "alignment.fqdump", "alignment.fastp",
        "alignment.STAR", "alignment.featureCount",
    ],
)
def bulk_rnaseq_pipeline(
    sra_ids: Optional[Union[str, Sequence[str]]] = None,
    samples: Optional[Union[
        Tuple[str, str, Optional[str]],
        Sequence[Tuple[str, str, Optional[str]]],
    ]] = None,
    genome_dir: str = "star_index",
    gtf: str = "genes.gtf",
    output_dir: str = "pipeline_output",
    genome_fasta_files: Optional[List[str]] = None,
    threads: int = 8,
    memory: str = "50G",
    jobs: Optional[int] = None,
    skip_download: bool = False,
    skip_qc: bool = False,
    gzip_fastq: bool = True,
    gene_mapping: bool = True,
    auto_install: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run a complete bulk RNA-seq pipeline from SRA accessions or local FASTQs.

    The pipeline chains: ``prefetch`` -> ``fqdump`` -> ``fastp`` -> ``STAR`` -> ``featureCount``.

    Parameters
    ----------
    sra_ids : str or list of str, optional
        SRA accession IDs to download.  Required unless *samples* is provided.
    samples : tuple or list of tuples, optional
        Pre-existing FASTQ sample tuples ``(name, fq1, fq2_or_None)``.
        When provided, the download step is skipped automatically.
    genome_dir : str
        Path to (or for) the STAR genome index directory.
    gtf : str
        Path to the GTF annotation file.
    output_dir : str
        Root output directory.  Sub-directories are created per step.
    genome_fasta_files : list of str, optional
        Genome FASTA file(s) for auto-building the STAR index.
    threads : int
        Threads per tool invocation.
    memory : str
        Memory limit for STAR BAM sorting (e.g. ``'50G'``).
    jobs : int, optional
        Number of concurrent jobs.  ``None`` auto-detects.
    skip_download : bool
        Skip the prefetch + fqdump steps (requires *samples*).
    skip_qc : bool
        Skip the fastp QC step.
    gzip_fastq : bool
        Compress FASTQ output from fqdump.
    gene_mapping : bool
        Map gene_id to gene_name in featureCounts output.
    auto_install : bool
        Auto-install missing CLI tools via conda/mamba.
    overwrite : bool
        Force re-run even when outputs already exist.

    Returns
    -------
    pandas.DataFrame
        Merged gene-level count matrix (genes x samples).
    """
    if sra_ids is None and samples is None:
        raise ValueError("Provide either sra_ids or samples.")

    prefetch_dir = os.path.join(output_dir, "prefetch")
    fastq_dir = os.path.join(output_dir, "fastq")
    fastp_dir = os.path.join(output_dir, "fastp")
    star_dir = os.path.join(output_dir, "star")
    counts_dir = os.path.join(output_dir, "counts")

    # ── Step 1: Download ──────────────────────────────────────────────
    if samples is not None:
        skip_download = True

    if not skip_download:
        assert sra_ids is not None
        ids = [sra_ids] if isinstance(sra_ids, str) else list(sra_ids)

        print(f"[pipeline] Step 1/4: Downloading {len(ids)} SRA accession(s)...", flush=True)
        prefetch(ids, output_dir=prefetch_dir, jobs=jobs, auto_install=auto_install)
        fq_results = fqdump(
            ids, output_dir=fastq_dir, sra_dir=prefetch_dir,
            gzip=gzip_fastq, threads=threads, jobs=jobs,
            auto_install=auto_install, force=overwrite,
        )
        fq_list = fq_results if isinstance(fq_results, list) else [fq_results]
        samples = [
            (r["srr"], r["fq1"], r.get("fq2"))
            for r in fq_list
        ]
    else:
        print("[pipeline] Step 1/4: Download skipped (samples provided).", flush=True)

    # Normalise samples to list
    assert samples is not None
    if isinstance(samples, tuple) and isinstance(samples[0], str):
        samples = [samples]  # type: ignore[list-item]
    sample_list: List[Tuple[str, str, Optional[str]]] = list(samples)  # type: ignore[arg-type]

    # ── Step 2: QC ────────────────────────────────────────────────────
    if not skip_qc:
        print(f"[pipeline] Step 2/4: Running fastp QC on {len(sample_list)} sample(s)...", flush=True)
        qc_results = fastp(
            sample_list, output_dir=fastp_dir, threads=threads,
            jobs=jobs, auto_install=auto_install, overwrite=overwrite,
        )
        qc_list = qc_results if isinstance(qc_results, list) else [qc_results]
        star_samples: List[Tuple[str, str, Optional[str]]] = [
            (r["sample"], r["clean1"], r.get("clean2"))
            for r in qc_list
        ]
    else:
        print("[pipeline] Step 2/4: QC skipped.", flush=True)
        star_samples = sample_list

    # ── Step 3: Alignment ─────────────────────────────────────────────
    print(f"[pipeline] Step 3/4: STAR alignment of {len(star_samples)} sample(s)...", flush=True)
    bam_results = STAR(
        star_samples, genome_dir=genome_dir, output_dir=star_dir,
        gtf=gtf, genome_fasta_files=genome_fasta_files,
        auto_index=True, threads=threads, memory=memory,
        jobs=jobs, auto_install=auto_install, overwrite=overwrite,
    )
    bam_list = bam_results if isinstance(bam_results, list) else [bam_results]
    bam_items: List[Tuple[str, str]] = [
        (r["sample"], r["bam"])
        for r in bam_list if "bam" in r
    ]

    if not bam_items:
        raise RuntimeError("No BAM files produced by STAR alignment.")

    # ── Step 4: Quantification ────────────────────────────────────────
    print(f"[pipeline] Step 4/4: featureCounts on {len(bam_items)} BAM file(s)...", flush=True)
    count_matrix = featureCount(
        bam_items, gtf=gtf, output_dir=counts_dir,
        gene_mapping=gene_mapping, merge_matrix=True,
        threads=threads, jobs=jobs, auto_install=auto_install,
        overwrite=overwrite,
    )

    print(f"[pipeline] Pipeline complete! Count matrix shape: {count_matrix.shape}", flush=True)
    return count_matrix
