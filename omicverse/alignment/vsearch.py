"""VSEARCH wrappers for 16S amplicon analysis.

Each function wraps one real ``vsearch`` subcommand
(https://github.com/torognes/vsearch). Install via
``conda install -c bioconda vsearch``.

Submodule-style layout (like ``ov.alignment.kb_api``) — each function name
matches the vsearch subcommand verb:

- :func:`merge_pairs`       — ``--fastq_mergepairs``
- :func:`filter_quality`    — ``--fastq_filter``
- :func:`dereplicate`       — ``--derep_fulllength``
- :func:`unoise3`           — ``--cluster_unoise`` (UNOISE3 denoising)
- :func:`uchime3_denovo`    — ``--uchime3_denovo`` (de novo chimera removal)
- :func:`sintax`            — ``--sintax`` (taxonomy assignment)
- :func:`usearch_global`    — ``--usearch_global --otutabout`` (count matrix)

No function writes to ``$HOME``; reference DB paths must be supplied
explicitly via ``db_fasta``.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .._registry import register_function
from ._cli_utils import (
    build_env,
    ensure_dir,
    resolve_executable,
    resolve_jobs,
    run_in_threads,
)


_Sample = Tuple[str, str, Optional[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vsearch_bin(explicit: Optional[str], auto_install: bool) -> Tuple[str, dict]:
    exe = resolve_executable("vsearch", explicit, auto_install=auto_install)
    env = build_env(extra_paths=[str(Path(exe).parent)])
    return exe, env


# ---------------------------------------------------------------------------
# merge_pairs
# ---------------------------------------------------------------------------


@register_function(
    aliases=["vsearch.merge_pairs", "fastq_mergepairs", "merge_paired_reads"],
    category="alignment",
    description="Merge paired-end FASTQs with vsearch --fastq_mergepairs.",
    examples=[
        "ov.alignment.vsearch.merge_pairs([('S1','S1_R1.fq.gz','S1_R2.fq.gz')], "
        "output_dir='run1/merged')",
    ],
    related=["alignment.cutadapt", "alignment.vsearch"],
)
def merge_pairs(
    samples: Union[_Sample, Sequence[_Sample]],
    output_dir: str,
    max_diffs: int = 10,
    min_overlap: int = 16,
    max_ns: int = 0,
    min_merge_len: int = 0,
    max_merge_len: int = 0,
    threads: int = 4,
    jobs: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
    vsearch_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> List[Dict[str, str]]:
    """Merge paired-end reads per sample.

    Parameters
    ----------
    samples
        ``(sample, fq1, fq2)`` tuple or list. ``fq2`` must be non-None.
    output_dir
        Directory for merged FASTQs (one per sample: ``<sample>_merged.fq``).
    max_diffs
        ``--fastq_maxdiffs`` (default 10).
    min_overlap
        ``--fastq_minovlen`` (default 16).
    max_ns
        ``--fastq_maxns`` (drop merged reads with more N bases).
    min_merge_len, max_merge_len
        ``--fastq_minmergelen`` / ``--fastq_maxmergelen``; 0 means unset.
    """
    out_root = ensure_dir(output_dir)
    vbin, env = _vsearch_bin(vsearch_path, auto_install)

    if isinstance(samples, tuple) and len(samples) == 3:
        sample_list: List[_Sample] = [samples]
    else:
        sample_list = list(samples)  # type: ignore[assignment]

    for sample, fq1, fq2 in sample_list:
        if not fq2:
            raise ValueError(
                f"merge_pairs requires paired-end input; sample '{sample}' has no fq2."
            )

    worker_count = resolve_jobs(len(sample_list), jobs, None)

    def _worker(item: _Sample) -> Dict[str, str]:
        sample, fq1, fq2 = item
        sample_dir = ensure_dir(out_root / sample)
        merged = sample_dir / f"{sample}_merged.fastq"
        log = sample_dir / f"{sample}.merge.log"

        if not overwrite and merged.exists() and merged.stat().st_size > 0:
            return {"sample": sample, "merged": str(merged), "log": str(log)}

        cmd = [
            vbin,
            "--fastq_mergepairs", str(fq1),
            "--reverse", str(fq2),
            "--fastqout", str(merged),
            "--fastq_maxdiffs", str(max_diffs),
            "--fastq_minovlen", str(min_overlap),
            "--fastq_maxns", str(max_ns),
            "--threads", str(threads),
        ]
        if min_merge_len > 0:
            cmd.extend(["--fastq_minmergelen", str(min_merge_len)])
        if max_merge_len > 0:
            cmd.extend(["--fastq_maxmergelen", str(max_merge_len)])
        if extra_args:
            cmd.extend(str(a) for a in extra_args)

        print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True)
        if proc.returncode != 0 or not merged.exists() or merged.stat().st_size == 0:
            raise RuntimeError(f"vsearch merge_pairs failed for {sample}; see {log}")
        return {"sample": sample, "merged": str(merged), "log": str(log)}

    return run_in_threads(sample_list, _worker, worker_count)


# ---------------------------------------------------------------------------
# filter_quality
# ---------------------------------------------------------------------------


@register_function(
    aliases=["vsearch.filter_quality", "fastq_filter"],
    category="alignment",
    description="Quality-filter merged FASTQ to FASTA with vsearch --fastq_filter, adding sample labels.",
    examples=[
        "ov.alignment.vsearch.filter_quality(merged_results, output_dir='run1/filtered', max_ee=1.0)",
    ],
    related=["alignment.vsearch"],
)
def filter_quality(
    merged: Union[Sequence[Dict[str, str]], Sequence[str], Sequence[Tuple[str, str]]],
    output_dir: str,
    max_ee: float = 1.0,
    min_len: int = 0,
    max_len: int = 0,
    trunc_len: int = 0,
    max_ns: int = 0,
    threads: int = 4,
    jobs: Optional[int] = None,
    extra_args: Optional[Sequence[str]] = None,
    vsearch_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> List[Dict[str, str]]:
    """Filter merged FASTQs and write per-sample FASTA with labels.

    Accepts:
      * list of dicts from :func:`merge_pairs` with keys ``sample`` and ``merged``
      * list of ``(sample, merged_path)`` tuples
      * list of fastq paths (sample name derived from filename)

    Output per sample: ``output_dir/<sample>/<sample>_filt.fasta`` with
    headers relabeled ``<sample>.<n>`` so downstream ``--otutabout`` can
    resolve sample identity from the read label prefix.
    """
    out_root = ensure_dir(output_dir)
    vbin, env = _vsearch_bin(vsearch_path, auto_install)

    normalized: List[Tuple[str, str]] = []
    for item in merged:
        if isinstance(item, dict):
            normalized.append((item["sample"], item["merged"]))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            normalized.append((str(item[0]), str(item[1])))
        elif isinstance(item, str):
            path = Path(item)
            name = path.name
            for sfx in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
                if name.endswith(sfx):
                    name = name[: -len(sfx)]
                    break
            if name.endswith("_merged"):
                name = name[: -len("_merged")]
            normalized.append((name, item))
        else:
            raise ValueError(f"Unrecognised input: {item!r}")

    worker_count = resolve_jobs(len(normalized), jobs, None)

    def _worker(item: Tuple[str, str]) -> Dict[str, str]:
        sample, merged_path = item
        sample_dir = ensure_dir(out_root / sample)
        filt = sample_dir / f"{sample}_filt.fasta"
        log = sample_dir / f"{sample}.filter.log"

        if not overwrite and filt.exists() and filt.stat().st_size > 0:
            return {"sample": sample, "filt": str(filt), "log": str(log)}

        cmd = [
            vbin,
            "--fastq_filter", str(merged_path),
            "--fastaout", str(filt),
            "--fastq_maxee", str(max_ee),
            "--fastq_maxns", str(max_ns),
            "--relabel", f"{sample}.",
            "--threads", str(threads),
        ]
        if min_len > 0:
            cmd.extend(["--fastq_minlen", str(min_len)])
        if max_len > 0:
            cmd.extend(["--fastq_maxlen", str(max_len)])
        if trunc_len > 0:
            cmd.extend(["--fastq_trunclen", str(trunc_len)])
        if extra_args:
            cmd.extend(str(a) for a in extra_args)

        print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True)
        if proc.returncode != 0 or not filt.exists() or filt.stat().st_size == 0:
            raise RuntimeError(f"vsearch filter_quality failed for {sample}; see {log}")
        return {"sample": sample, "filt": str(filt), "log": str(log)}

    return run_in_threads(normalized, _worker, worker_count)


# ---------------------------------------------------------------------------
# dereplicate
# ---------------------------------------------------------------------------


@register_function(
    aliases=["vsearch.dereplicate", "derep_fulllength"],
    category="alignment",
    description="Concatenate per-sample filtered FASTAs and dereplicate with vsearch --derep_fulllength.",
    examples=[
        "ov.alignment.vsearch.dereplicate(filt_results, output_dir='run1/derep', min_uniq=2)",
    ],
    related=["alignment.vsearch"],
)
def dereplicate(
    filtered: Union[Sequence[Dict[str, str]], Sequence[str]],
    output_dir: str,
    min_uniq: int = 2,
    threads: int = 4,
    extra_args: Optional[Sequence[str]] = None,
    vsearch_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Combine + dereplicate filtered FASTAs → one ``uniques.fasta``.

    The concatenated file ``combined.fasta`` is also written; downstream
    :func:`usearch_global` uses it to build the sample × ASV count matrix
    because it preserves per-read sample labels.
    """
    out_root = ensure_dir(output_dir)
    vbin, env = _vsearch_bin(vsearch_path, auto_install)

    inputs: List[str] = []
    for item in filtered:
        if isinstance(item, dict):
            inputs.append(item["filt"])
        else:
            inputs.append(str(item))

    combined = out_root / "combined.fasta"
    uniques = out_root / "uniques.fasta"
    log = out_root / "derep.log"

    if not overwrite and uniques.exists() and uniques.stat().st_size > 0 and combined.exists():
        return {
            "combined": str(combined),
            "uniques": str(uniques),
            "log": str(log),
        }

    with open(combined, "wb") as dst:
        for path in inputs:
            src_path = Path(path)
            with open(src_path, "rb") as src:
                shutil.copyfileobj(src, dst)

    cmd = [
        vbin,
        "--derep_fulllength", str(combined),
        "--output", str(uniques),
        "--sizein", "--sizeout",
        "--minuniquesize", str(min_uniq),
        "--strand", "plus",
        "--threads", str(threads),
    ]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True)
    if proc.returncode != 0 or not uniques.exists() or uniques.stat().st_size == 0:
        raise RuntimeError(f"vsearch dereplicate failed; see {log}")

    return {"combined": str(combined), "uniques": str(uniques), "log": str(log)}


# ---------------------------------------------------------------------------
# unoise3
# ---------------------------------------------------------------------------


@register_function(
    aliases=["vsearch.unoise3", "cluster_unoise", "denoise_asv"],
    category="alignment",
    description="Denoise unique sequences into ASVs with vsearch --cluster_unoise (UNOISE3).",
    examples=[
        "ov.alignment.vsearch.unoise3('run1/derep/uniques.fasta', output_dir='run1/asv', minsize=2)",
    ],
    related=["alignment.vsearch"],
)
def unoise3(
    uniques_fasta: str,
    output_dir: str,
    alpha: float = 2.0,
    minsize: int = 2,
    threads: int = 4,
    extra_args: Optional[Sequence[str]] = None,
    vsearch_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Run UNOISE3 denoising to build ASVs (amplicon sequence variants).

    Equivalent biological resolution to DADA2 ASVs per multiple benchmarks
    (Vestergaard 2024 ISME Comms); uses VSEARCH's C implementation of the
    UNOISE3 algorithm (``--cluster_unoise``).

    Output
    ------
    ``asvs_pre.fasta`` — raw ASV centroids (pre-chimera removal).
    """
    out_root = ensure_dir(output_dir)
    vbin, env = _vsearch_bin(vsearch_path, auto_install)

    asv = out_root / "asvs_pre.fasta"
    log = out_root / "unoise3.log"

    if not overwrite and asv.exists() and asv.stat().st_size > 0:
        return {"asv": str(asv), "log": str(log)}

    cmd = [
        vbin,
        "--cluster_unoise", str(uniques_fasta),
        "--centroids", str(asv),
        "--minsize", str(minsize),
        "--unoise_alpha", str(alpha),
        "--sizein", "--sizeout",
        "--threads", str(threads),
    ]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True)
    if proc.returncode != 0 or not asv.exists() or asv.stat().st_size == 0:
        raise RuntimeError(f"vsearch unoise3 failed; see {log}")

    return {"asv": str(asv), "log": str(log)}


# ---------------------------------------------------------------------------
# uchime3_denovo
# ---------------------------------------------------------------------------


@register_function(
    aliases=["vsearch.uchime3_denovo", "uchime3", "chimera_removal"],
    category="alignment",
    description="Remove chimeric ASVs de novo with vsearch --uchime3_denovo.",
    examples=[
        "ov.alignment.vsearch.uchime3_denovo('run1/asv/asvs_pre.fasta', output_dir='run1/asv')",
    ],
    related=["alignment.vsearch"],
)
def uchime3_denovo(
    asvs_fasta: str,
    output_dir: str,
    vsearch_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Dict[str, str]:
    """De novo chimera detection / removal on UNOISE3 ASVs.

    Note that UNOISE3 already applies a lightweight chimera filter; running
    ``uchime3_denovo`` as a second pass is a conservative extra step.

    .. note::
        vsearch's ``--uchime3_denovo`` is intentionally **single-threaded**
        upstream, so there is no ``threads=`` parameter (unlike the other
        wrappers in this module).
    """
    out_root = ensure_dir(output_dir)
    vbin, env = _vsearch_bin(vsearch_path, auto_install)

    nonchim = out_root / "asvs.fasta"
    chim = out_root / "chimeras.fasta"
    log = out_root / "uchime3.log"

    if not overwrite and nonchim.exists() and nonchim.stat().st_size > 0:
        return {"asv": str(nonchim), "chimeras": str(chim), "log": str(log)}

    cmd = [
        vbin,
        "--uchime3_denovo", str(asvs_fasta),
        "--nonchimeras", str(nonchim),
        "--chimeras", str(chim),
        "--sizein", "--sizeout",
    ]

    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True)
    if (proc.returncode != 0
            or not nonchim.exists()
            or nonchim.stat().st_size == 0):
        raise RuntimeError(
            f"vsearch uchime3_denovo failed or produced empty non-chimera "
            f"output (all ASVs classified as chimeric?); see {log}"
        )

    return {"asv": str(nonchim), "chimeras": str(chim), "log": str(log)}


# ---------------------------------------------------------------------------
# sintax
# ---------------------------------------------------------------------------


@register_function(
    aliases=["vsearch.sintax", "sintax", "taxonomy_sintax"],
    category="alignment",
    description="Assign taxonomy to ASVs with vsearch --sintax against a SINTAX-formatted reference.",
    examples=[
        "ov.alignment.vsearch.sintax('run1/asv/asvs.fasta', "
        "db_fasta='/scratch/.../db/rdp/rdp_16s_v18.fa.gz', "
        "output_dir='run1/tax', cutoff=0.8)",
    ],
    related=["alignment.vsearch", "alignment.fetch_silva"],
)
def sintax(
    asvs_fasta: str,
    db_fasta: str,
    output_dir: str,
    cutoff: float = 0.8,
    strand: str = "both",
    threads: int = 4,
    extra_args: Optional[Sequence[str]] = None,
    vsearch_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Taxonomy assignment via SINTAX.

    Parameters
    ----------
    asvs_fasta
        Query ASV FASTA.
    db_fasta
        **Required** path to a SINTAX-formatted reference FASTA (headers must
        encode taxonomy as ``;tax=d:...,p:...,c:...;``). Must be explicitly
        provided — no ``$HOME`` fallback. See :func:`omicverse.alignment.fetch_silva`.
    cutoff
        Bootstrap confidence threshold (default 0.8). Stored in SINTAX output
        as two columns: raw classifications and cutoff-filtered classifications.
    strand
        ``plus`` | ``minus`` | ``both``.
    """
    if not db_fasta:
        raise ValueError(
            "sintax requires an explicit db_fasta (SINTAX-formatted reference). "
            "omicverse never writes reference DBs to $HOME — point db_fasta at "
            "a path under /scratch (or use ov.alignment.fetch_silva())."
        )
    db_path = Path(db_fasta)
    if not db_path.exists():
        raise FileNotFoundError(f"SINTAX reference not found: {db_fasta}")

    out_root = ensure_dir(output_dir)
    vbin, env = _vsearch_bin(vsearch_path, auto_install)

    tabbed = out_root / "sintax.tsv"
    log = out_root / "sintax.log"

    if not overwrite and tabbed.exists() and tabbed.stat().st_size > 0:
        return {"tsv": str(tabbed), "log": str(log)}

    cmd = [
        vbin,
        "--sintax", str(asvs_fasta),
        "--db", str(db_path),
        "--tabbedout", str(tabbed),
        "--sintax_cutoff", str(cutoff),
        "--strand", strand,
        "--threads", str(threads),
    ]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True)
    if proc.returncode != 0 or not tabbed.exists():
        raise RuntimeError(f"vsearch sintax failed; see {log}")

    return {"tsv": str(tabbed), "log": str(log)}


# ---------------------------------------------------------------------------
# usearch_global  (for sample × ASV count matrix)
# ---------------------------------------------------------------------------


@register_function(
    aliases=["vsearch.usearch_global", "otutab", "count_matrix"],
    category="alignment",
    description="Map per-sample reads back to ASVs with vsearch --usearch_global --otutabout to produce the sample × ASV count matrix.",
    examples=[
        "ov.alignment.vsearch.usearch_global('run1/derep/combined.fasta', "
        "'run1/asv/asvs.fasta', output_dir='run1/otutab')",
    ],
    related=["alignment.vsearch"],
)
def usearch_global(
    reads_fasta: str,
    asvs_fasta: str,
    output_dir: str,
    identity: float = 0.97,
    threads: int = 4,
    strand: str = "plus",
    extra_args: Optional[Sequence[str]] = None,
    vsearch_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Build the ASV count matrix.

    ``reads_fasta`` is typically the concatenated filtered FASTA (with
    ``<sample>.<n>`` headers) from :func:`dereplicate` (output ``combined``).
    Each read is mapped to the closest ASV at ``identity`` (default 0.97) and
    tallied per sample.

    Output
    ------
    ``otutab.tsv`` — tab-delimited; first column ``#OTU ID``, subsequent
    columns one per sample, rows = ASVs.
    """
    out_root = ensure_dir(output_dir)
    vbin, env = _vsearch_bin(vsearch_path, auto_install)

    otutab = out_root / "otutab.tsv"
    log = out_root / "usearch_global.log"

    if not overwrite and otutab.exists() and otutab.stat().st_size > 0:
        return {"otutab": str(otutab), "log": str(log)}

    cmd = [
        vbin,
        "--usearch_global", str(reads_fasta),
        "--db", str(asvs_fasta),
        "--id", str(identity),
        "--strand", strand,
        "--otutabout", str(otutab),
        "--threads", str(threads),
    ]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True)
    if proc.returncode != 0 or not otutab.exists():
        raise RuntimeError(f"vsearch usearch_global failed; see {log}")

    return {"otutab": str(otutab), "log": str(log)}
