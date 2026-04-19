"""cutadapt wrapper for amplicon primer trimming.

Wraps the real ``cutadapt`` CLI (https://cutadapt.readthedocs.io) —
install via ``pip install cutadapt`` or ``conda install -c bioconda cutadapt``.

The function follows the same shape as :func:`omicverse.alignment.fastp`:
takes ``(sample, fq1, fq2)`` tuples, writes trimmed FASTQs into per-sample
subdirectories under ``output_dir``, and returns paths. No implicit writes
to ``$HOME``.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .._registry import register_function
from ._cli_utils import (
    build_env,
    ensure_dir,
    is_gz,
    resolve_executable,
    resolve_jobs,
    run_in_threads,
)


def _out_ext(fq1: Path, force_gzip: Optional[bool]) -> str:
    if force_gzip is None:
        return ".fastq.gz" if is_gz(fq1) else ".fastq"
    return ".fastq.gz" if force_gzip else ".fastq"


def _derive_sample_name(fq1: Path) -> str:
    name = fq1.name
    for suffix in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for tag in ("_1", "_2", "_R1", "_R2", "_R1_001", "_R2_001"):
        if name.endswith(tag):
            name = name[: -len(tag)]
            break
    return name


def _normalize_samples(
    samples: Union[
        Tuple[str, str, Optional[str]],
        Sequence[Tuple[str, str, Optional[str]]],
        Sequence[str],
    ]
) -> Tuple[List[Tuple[str, str, Optional[str]]], bool]:
    if isinstance(samples, tuple) and len(samples) == 3:
        return [samples], True
    if isinstance(samples, (list, tuple)) and samples and isinstance(samples[0], str):
        if len(samples) == 1:
            fq1 = Path(samples[0])
            return [(_derive_sample_name(fq1), str(fq1), None)], True
        if len(samples) == 2:
            fq1 = Path(samples[0])
            fq2 = Path(samples[1])
            return [(_derive_sample_name(fq1), str(fq1), str(fq2))], True
        raise ValueError("When passing a list of strings, provide 1 or 2 FASTQ paths.")
    return list(samples), False  # type: ignore[arg-type]


def _run_cutadapt_one(
    sample: str,
    fq1: Path,
    fq2: Optional[Path],
    out_root: Path,
    primer_fwd: str,
    primer_rev: Optional[str],
    threads: int,
    force_gzip: Optional[bool],
    discard_untrimmed: bool,
    min_length: int,
    max_n: Optional[int],
    extra_args: Optional[Sequence[str]],
    cutadapt_bin: str,
    env: dict,
    overwrite: bool,
) -> Dict[str, str]:
    sample_dir = ensure_dir(out_root / sample)
    ext = _out_ext(fq1, force_gzip)
    trim1 = sample_dir / f"{sample}_trim_1{ext}"
    trim2 = sample_dir / f"{sample}_trim_2{ext}" if fq2 else None
    log = sample_dir / f"{sample}.cutadapt.log"

    if not overwrite:
        if trim1.exists() and trim1.stat().st_size > 0 and log.exists():
            if not fq2 or (trim2 and trim2.exists() and trim2.stat().st_size > 0):
                return {
                    "sample": sample,
                    "trim1": str(trim1),
                    "trim2": str(trim2) if trim2 else "",
                    "log": str(log),
                }

    cmd = [cutadapt_bin, "-j", str(threads), "-g", primer_fwd]
    if fq2 and primer_rev:
        cmd.extend(["-G", primer_rev])
    if discard_untrimmed:
        cmd.append("--discard-untrimmed")
    if min_length and min_length > 0:
        cmd.extend(["--minimum-length", str(min_length)])
    if max_n is not None:
        cmd.extend(["--max-n", str(max_n)])
    cmd.extend(["-o", str(trim1)])
    if fq2:
        cmd.extend(["-p", str(trim2)])
    cmd.append(str(fq1))
    if fq2:
        cmd.append(str(fq2))
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    with open(log, "w") as fh:
        print(">>", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"cutadapt failed (exit {proc.returncode}) — see {log}"
            )

    if not (trim1.exists() and trim1.stat().st_size > 0):
        raise RuntimeError(f"cutadapt produced no output at {trim1}")

    return {
        "sample": sample,
        "trim1": str(trim1),
        "trim2": str(trim2) if trim2 else "",
        "log": str(log),
    }


@register_function(
    aliases=["cutadapt", "primer_trim", "16s_primer_trim"],
    category="alignment",
    description="Trim 16S/ITS/amplicon PCR primers with cutadapt (paired-end or single-end).",
    examples=[
        "ov.alignment.cutadapt([('S1','S1_R1.fq.gz','S1_R2.fq.gz')], "
        "primer_fwd='GTGYCAGCMGCCGCGGTAA', primer_rev='GGACTACNVGGGTWTCTAAT', "
        "output_dir='run1/cutadapt')",
    ],
    related=["alignment.vsearch", "alignment.amplicon_16s_pipeline"],
)
def cutadapt(
    samples: Union[
        Tuple[str, str, Optional[str]], Sequence[Tuple[str, str, Optional[str]]]
    ],
    primer_fwd: str,
    primer_rev: Optional[str] = None,
    output_dir: str = "cutadapt",
    threads: int = 4,
    jobs: Optional[int] = None,
    output_gzip: Optional[bool] = None,
    discard_untrimmed: bool = True,
    min_length: int = 50,
    max_n: Optional[int] = 0,
    extra_args: Optional[Sequence[str]] = None,
    cutadapt_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """Run cutadapt to remove amplicon PCR primers.

    Parameters
    ----------
    samples
        ``(sample, fq1, fq2)`` tuple, list of such tuples, or 1-2 FASTQ paths.
    primer_fwd
        Forward primer sequence (5' anchor on R1). IUPAC ambiguity allowed.
        Common 16S V4 choice: ``GTGYCAGCMGCCGCGGTAA`` (515F Parada).
    primer_rev
        Reverse primer (5' anchor on R2). For V4: ``GGACTACNVGGGTWTCTAAT`` (806R Apprill).
    output_dir
        Output directory; per-sample subdirs are created under it.
    threads
        Threads per cutadapt invocation.
    jobs
        Concurrent sample jobs (default: CPU/2, capped by sample count).
    output_gzip
        Force gzipped output; ``None`` follows input suffix.
    discard_untrimmed
        Drop read pairs where primers were not found (standard for 16S).
    min_length
        Minimum post-trim length; pairs shorter than this are dropped.
    max_n
        Maximum ambiguous bases allowed per read (``None`` disables filter).
    extra_args
        Additional cutadapt CLI arguments appended verbatim.
    cutadapt_path
        Explicit path to ``cutadapt`` executable.
    auto_install
        Try to install via conda when missing.
    overwrite
        Re-run and overwrite existing outputs.
    """
    sample_list, single_input = _normalize_samples(samples)

    out_root = ensure_dir(output_dir)
    cutadapt_bin = resolve_executable("cutadapt", cutadapt_path, auto_install=auto_install)
    env = build_env(extra_paths=[str(Path(cutadapt_bin).parent)])

    worker_count = resolve_jobs(len(sample_list), jobs, None)

    def _worker(item: Tuple[str, str, Optional[str]]) -> Dict[str, str]:
        sample, fq1, fq2 = item
        return _run_cutadapt_one(
            sample=sample,
            fq1=Path(fq1),
            fq2=Path(fq2) if fq2 else None,
            out_root=out_root,
            primer_fwd=primer_fwd,
            primer_rev=primer_rev,
            threads=threads,
            force_gzip=output_gzip,
            discard_untrimmed=discard_untrimmed,
            min_length=min_length,
            max_n=max_n,
            extra_args=extra_args,
            cutadapt_bin=cutadapt_bin,
            env=env,
            overwrite=overwrite,
        )

    results = run_in_threads(sample_list, _worker, worker_count)
    if single_input:
        return results[0]
    return results
