"""fasterq-dump wrapper for SRA -> FASTQ conversion."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .._registry import register_function
from ._cli_utils import (
    build_env,
    ensure_dir,
    find_sra_file,
    listify,
    pick_compressor,
    resolve_executable,
    resolve_jobs,
    run_cmd,
    run_in_threads,
)


def _detect_outputs(out_dir: Path, srr: str, gzip_output: bool) -> Tuple[Optional[Path], Optional[Path]]:
    ext = ".fastq.gz" if gzip_output else ".fastq"
    prefixes = [srr, f"{srr}.sra", f"{srr}.sralite"]
    for prefix in prefixes:
        fq1 = out_dir / f"{prefix}_1{ext}"
        fq2 = out_dir / f"{prefix}_2{ext}"
        if fq1.exists() and fq1.stat().st_size > 0:
            if fq2.exists() and fq2.stat().st_size > 0:
                return fq1, fq2
            return fq1, None
        single = out_dir / f"{prefix}{ext}"
        if single.exists() and single.stat().st_size > 0:
            return single, None
    return None, None


def _compress_fastq(path: Path) -> Path:
    if path.suffix == ".gz":
        return path
    exe, args = pick_compressor()
    run_cmd([exe, *args, str(path)])
    gz_path = path.with_suffix(path.suffix + ".gz")
    if not gz_path.exists() or gz_path.stat().st_size == 0:
        raise RuntimeError(f"Compression failed for {path}")
    return gz_path


def _normalize_layout(layout: str) -> str:
    layout = layout.lower()
    if layout not in {"auto", "single", "paired"}:
        raise ValueError("library_layout must be 'auto', 'single', or 'paired'")
    return layout


@register_function(
    aliases=["fqdump", "fasterq", "fasterq-dump"],
    category="alignment",
    description="Convert SRA accessions to FASTQ using fasterq-dump.",
    examples=[
        "ov.alignment.fqdump('SRR123', output_dir='fastq', threads=8)",
        "ov.alignment.fqdump(['SRR1','SRR2'], output_dir='fastq', gzip=True)",
    ],
    related=["alignment.prefetch", "alignment.fastp", "alignment.STAR", "alignment.featureCount"],
)
def fqdump(
    sra_ids: Union[str, Sequence[str]],
    output_dir: str = "fastq",
    threads: int = 8,
    memory: str = "4G",
    temp_dir: Optional[str] = None,
    gzip: bool = False,
    library_layout: str = "auto",
    jobs: Optional[int] = None,
    max_workers: Optional[int] = None,
    retries: int = 2,
    sra_dir: Optional[str] = None,
    fasterq_path: Optional[str] = None,
    auto_install: bool = True,
    force: bool = False,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """
    Convert SRA accessions to FASTQ.

    Parameters
    ----------
    sra_ids
        SRR accession (str) or list of accessions.
    output_dir
        Output directory for FASTQ files (per-sample subdir).
    threads
        Threads per fasterq-dump job.
    memory
        Memory limit passed to --mem (e.g. '4G').
    temp_dir
        Temporary directory root for fasterq-dump.
    gzip
        Compress FASTQ outputs with pigz/gzip.
    library_layout
        'auto', 'single', or 'paired'.
    jobs
        Concurrent jobs.
    max_workers
        Legacy alias for jobs.
    retries
        Retries per accession.
    sra_dir
        Directory containing prefetched .sra/.sralite files.
    fasterq_path
        Explicit path to fasterq-dump.
    force
        Force overwrite existing output files (adds --force).
    auto_install
        Install missing tools automatically when possible.
    """
    srrs = listify(sra_ids)
    out_root = ensure_dir(output_dir)
    layout = _normalize_layout(library_layout)

    worker_count = resolve_jobs(len(srrs), jobs, max_workers)

    fasterq_bin = resolve_executable("fasterq-dump", fasterq_path, auto_install=auto_install)
    env = build_env(extra_paths=[str(Path(fasterq_bin).parent)])

    def _select_input(srr: str) -> str:
        if sra_dir:
            sra_path = find_sra_file(srr, sra_dir)
            if sra_path:
                return str(sra_path)
        return srr

    def _run_one(srr: str) -> Dict[str, str]:
        sample_dir = ensure_dir(out_root / srr)

        fq1, fq2 = _detect_outputs(sample_dir, srr, gzip)
        if not fq1 and gzip:
            raw1, raw2 = _detect_outputs(sample_dir, srr, False)
            if raw1:
                raw1 = _compress_fastq(raw1)
                raw2 = _compress_fastq(raw2) if raw2 else None
                fq1, fq2 = raw1, raw2
        if fq1:
            detected = "paired" if fq2 else "single"
            if layout == "paired" and detected != "paired":
                raise RuntimeError(f"Expected paired-end output for {srr}, but only single-end found")
            if layout == "single" and detected != "single":
                raise RuntimeError(f"Expected single-end output for {srr}, but paired files found")
            return {"srr": srr, "fq1": str(fq1), "fq2": str(fq2) if fq2 else "", "layout": detected}

        input_token = _select_input(srr)

        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                if force:
                    prefixes = [srr, f"{srr}.sra", f"{srr}.sralite"]
                    for ext in (".fastq", ".fastq.gz"):
                        for suffix in ("_1", "_2", ""):
                            for prefix in prefixes:
                                candidate = sample_dir / f"{prefix}{suffix}{ext}"
                                if candidate.exists():
                                    candidate.unlink()
                cmd = [
                    fasterq_bin,
                    input_token,
                    "-O", str(sample_dir),
                    "-e", str(threads),
                    "--mem", str(memory),
                    "--split-files",
                    "--progress",
                ]
                if temp_dir:
                    cmd.extend(["-t", str(Path(temp_dir) / srr)])
                if force:
                    cmd.append("--force")
                run_cmd(cmd, env=env)

                fq1, fq2 = _detect_outputs(sample_dir, srr, False)
                if fq1 is None:
                    raise RuntimeError(f"fasterq-dump produced no FASTQ for {srr}")
                if gzip:
                    fq1 = _compress_fastq(fq1)
                    fq2 = _compress_fastq(fq2) if fq2 else None

                detected = "paired" if fq2 else "single"
                if layout == "paired" and detected != "paired":
                    raise RuntimeError(f"Expected paired-end output for {srr}, but only single-end found")
                if layout == "single" and detected != "single":
                    raise RuntimeError(f"Expected single-end output for {srr}, but paired files found")

                return {
                    "srr": srr,
                    "fq1": str(fq1),
                    "fq2": str(fq2) if fq2 else "",
                    "layout": detected,
                }
            except Exception as exc:
                last_err = exc
                time.sleep(min(5 * attempt, 20))

        raise RuntimeError(f"fasterq-dump failed for {srr}: {last_err}")

    results = run_in_threads(srrs, _run_one, worker_count)
    if isinstance(sra_ids, str):
        return results[0]
    return results
