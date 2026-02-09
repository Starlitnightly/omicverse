"""fastp wrapper for FASTQ quality control."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..utils.registry import register_function
from ._cli_utils import (
    build_env,
    ensure_dir,
    is_gz,
    resolve_executable,
    resolve_jobs,
    run_cmd,
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
    for tag in ("_1", "_2"):
        if name.endswith(tag):
            name = name[: -len(tag)]
            break
    if name.endswith(".sra"):
        name = name[: -4]
    if name.endswith(".sralite"):
        name = name[: -8]
    return name


def _normalize_samples(
    samples: Union[Tuple[str, str, Optional[str]], Sequence[Tuple[str, str, Optional[str]]], Sequence[str]]
) -> Tuple[List[Tuple[str, str, Optional[str]]], bool]:
    if isinstance(samples, tuple) and len(samples) == 3:
        return [samples], True

    if isinstance(samples, (list, tuple)) and samples and isinstance(samples[0], str):
        if len(samples) == 1:
            fq1 = Path(samples[0])
            sample = _derive_sample_name(fq1)
            return [(sample, str(fq1), None)], True
        if len(samples) == 2:
            fq1 = Path(samples[0])
            fq2 = Path(samples[1])
            sample = _derive_sample_name(fq1)
            return [(sample, str(fq1), str(fq2))], True
        raise ValueError("When passing a list of strings, provide 1 (single-end) or 2 (paired-end) FASTQs.")

    sample_list = list(samples)  # type: ignore[arg-type]
    return sample_list, False

def _run_fastp_one(
    sample: str,
    fq1: Path,
    fq2: Optional[Path],
    out_root: Path,
    threads: int,
    force_gzip: Optional[bool],
    extra_args: Optional[Sequence[str]],
    fastp_bin: str,
    env: dict,
    overwrite: bool,
) -> Dict[str, str]:
    sample_dir = ensure_dir(out_root / sample)
    ext = _out_ext(fq1, force_gzip)

    clean1 = sample_dir / f"{sample}_clean_1{ext}"
    clean2 = sample_dir / f"{sample}_clean_2{ext}" if fq2 else None
    json = sample_dir / f"{sample}.fastp.json"
    html = sample_dir / f"{sample}.fastp.html"

    if not overwrite:
        if clean1.exists() and clean1.stat().st_size > 0 and json.exists() and html.exists():
            if not fq2 or (clean2 and clean2.exists() and clean2.stat().st_size > 0):
                return {
                    "sample": sample,
                    "clean1": str(clean1),
                    "clean2": str(clean2) if clean2 else "",
                    "json": str(json),
                    "html": str(html),
                }
    else:
        for path in (clean1, clean2, json, html):
            if path and path.exists():
                path.unlink()

    cmd = [
        fastp_bin,
        "-i", str(fq1),
        "-o", str(clean1),
        "-w", str(threads),
        "-j", str(json),
        "-h", str(html),
    ]
    if fq2:
        cmd.extend(["-I", str(fq2), "-O", str(clean2), "--detect_adapter_for_pe"])
    if extra_args:
        cmd.extend(list(extra_args))

    run_cmd(cmd, env=env)

    if not (clean1.exists() and clean1.stat().st_size > 0):
        raise RuntimeError(f"fastp failed to produce {clean1}")
    if fq2 and not (clean2 and clean2.exists() and clean2.stat().st_size > 0):
        raise RuntimeError(f"fastp failed to produce {clean2}")

    return {
        "sample": sample,
        "clean1": str(clean1),
        "clean2": str(clean2) if clean2 else "",
        "json": str(json),
        "html": str(html),
    }


@register_function(
    aliases=["fastp", "qc_fastp"],
    category="alignment",
    description="Run fastp QC on FASTQ files (single-end or paired-end).",
    examples=[
        "ov.alignment.fastp([('S1','S1_1.fq.gz','S1_2.fq.gz')], output_dir='fastp')",
        "ov.alignment.fastp(('S1','S1.fq.gz', None), output_dir='fastp')",
    ],
    related=["alignment.fqdump", "alignment.STAR"],
)
def fastp(
    samples: Union[Tuple[str, str, Optional[str]], Sequence[Tuple[str, str, Optional[str]]]],
    output_dir: str = "fastp",
    threads: int = 8,
    jobs: Optional[int] = None,
    max_workers: Optional[int] = None,
    output_gzip: Optional[bool] = None,
    extra_args: Optional[Sequence[str]] = None,
    fastp_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """
    Run fastp QC.

    Parameters
    ----------
    samples
        (sample, fq1, fq2) tuple, list of such tuples, or a list of 1/2 FASTQ paths.
    output_dir
        Output directory for cleaned FASTQs (per-sample subdir).
    threads
        Threads per fastp job.
    jobs
        Concurrent jobs.
    max_workers
        Legacy alias for jobs.
    output_gzip
        Force output gzip; None follows input fq1 suffix.
    extra_args
        Additional fastp CLI arguments.
    fastp_path
        Explicit path to fastp executable.
    auto_install
        Install missing tools automatically when possible.
    overwrite
        If True, rerun fastp and overwrite existing outputs.
    """
    sample_list, single_input = _normalize_samples(samples)

    out_root = ensure_dir(output_dir)
    fastp_bin = resolve_executable("fastp", fastp_path, auto_install=auto_install)
    env = build_env(extra_paths=[str(Path(fastp_bin).parent)])

    worker_count = resolve_jobs(len(sample_list), jobs, max_workers)

    def _worker(item: Tuple[str, str, Optional[str]]) -> Dict[str, str]:
        sample, fq1, fq2 = item
        return _run_fastp_one(
            sample=sample,
            fq1=Path(fq1),
            fq2=Path(fq2) if fq2 else None,
            out_root=out_root,
            threads=threads,
            force_gzip=output_gzip,
            extra_args=extra_args,
            fastp_bin=fastp_bin,
            env=env,
            overwrite=overwrite,
        )

    results = run_in_threads(sample_list, _worker, worker_count)
    if single_input:
        return results[0]
    return results
