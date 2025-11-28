# qc_fastp.py fastp step factory
from __future__ import annotations

try:
    from .qc_tools import fastp_clean_parallel
except ImportError:
    from qc_tools import fastp_clean_parallel

import os, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, Tuple, List, Dict, Optional

try:
    from .tools_check import which_or_find, merged_env
except ImportError:
    from tools_check import which_or_find, merged_env


def _run(cmd: list[str], env: dict | None = None):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ext(p: Path) -> str:
    # Preserve input behaviour: emit .gz when the input is gzipped.
    return ".fastq.gz" if p.suffix == ".gz" or p.name.endswith(".fastq.gz") else ".fastq"

def _fastp_one(
    srr: str,
    fq1: Path,
    fq2: Optional[Path],
    out_root: Path,
    threads: int = 8,
) -> Tuple[str, Path, Optional[Path], Path, Path]:
    """
    Run fastp for a single sample (single-end or paired-end).
    Returns (srr, clean1, clean2, json, html).
    For single-end: clean2 will be None.
    """
    env = merged_env()
    fastp_bin = which_or_find("fastp")  # Resolve the executable path.
    out_dir = _ensure_dir(out_root / srr)

    # Compose output filenames (respect the input suffix behaviour).
    oext = _ext(fq1)
    clean1 = out_dir / f"{srr}_clean_1{oext.replace('.fastq', '')}"
    clean2 = out_dir / f"{srr}_clean_2{oext.replace('.fastq', '')}" if fq2 else None
    json = out_dir / f"{srr}.fastp.json"
    html = out_dir / f"{srr}.fastp.html"

    # Check if processing is needed
    if fq2:  # Paired-end
        outputs_exist = (clean1.exists() and clean2.exists() and json.exists() and html.exists())
    else:  # Single-end
        outputs_exist = (clean1.exists() and json.exists() and html.exists())

    if outputs_exist:
        return srr, clean1, clean2, json, html

    # Build command based on single-end vs paired-end
    if fq2:  # Paired-end processing
        cmd = [
            fastp_bin,
            "-i", str(fq1),
            "-I", str(fq2),
            "-o", str(clean1),
            "-O", str(clean2),
            "-w", str(threads),
            "-j", str(json),
            "-h", str(html),
            "--detect_adapter_for_pe",
            "--thread", str(threads),
            "--overrepresentation_analysis",
        ]
    else:  # Single-end processing
        cmd = [
            fastp_bin,
            "-i", str(fq1),
            "-o", str(clean1),
            "-w", str(threads),
            "-j", str(json),
            "-h", str(html),
            "--thread", str(threads),
            "--overrepresentation_analysis",
        ]

    # fastp compresses automatically when the suffix is .gz.
    # Optional stricter filtering parameters (quality, unqualified %, minimum length):
    # cmd += ["-q", "20", "-u", "30", "-l", "30"]

    _run(cmd, env=env)

    # Basic validation
    if fq2:  # Paired-end validation
        if not (clean1.exists() and clean1.stat().st_size > 0 and clean2.exists() and clean2.stat().st_size > 0):
            raise RuntimeError(f"fastp outputs missing/empty for {srr} in {out_dir}")
    else:  # Single-end validation
        if not (clean1.exists() and clean1.stat().st_size > 0):
            raise RuntimeError(f"fastp output missing/empty for {srr} in {out_dir}")

    return srr, clean1, clean2, json, html


def fastp_batch(
    pairs: Sequence[Tuple[str, Path, Optional[Path]]],  # [(srr, fq1,>Optional[fq2]), ...]
    out_root: str | Path,
    threads: int = 12,          # fastp threads per sample.
    max_workers: int | None = None,  # Number of samples processed concurrently; None lets us auto-select.
) -> List[Tuple[str, Path, Optional[Path], Path, Path]]:
    """
    Execute fastp concurrently for multiple samples.
    Supports both single-end (fq2=None) and paired-end processing.
    Returns [(srr, clean1, clean2, json, html), ...] in the original ordering.
    For single-end samples, clean2 will be None.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        # Default concurrency: min(threads, cpu_count/2); adjust to taste.
        import os, math
        max_workers = max(1, min(threads, (os.cpu_count() or 8) // 2))

    results: Dict[str, Tuple[str, Path, Optional[Path], Path, Path]] = {}
    errors: List[Tuple[str, str]] = []

    def _worker(item: Tuple[str, Path, Optional[Path]]):
        srr, fq1, fq2 = item
        return _fastp_one(srr, Path(fq1), Path(fq2) if fq2 else None, out_root=out_root, threads=threads)

    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futs = {ex.submit(_worker, it): it[0] for it in pairs}
        for fut in as_completed(futs):
            srr = futs[fut]
            try:
                ret = fut.result()
                results[srr] = ret
            except Exception as e:
                errors.append((srr, str(e)))

    if errors:
        msg = "; ".join([f"{s}:{m}" for s, m in errors])
        raise RuntimeError(f"fastp_batch failed for {len(errors)} samples: {msg}")

    # Preserve the original order.
    order = {s: i for i, (s, _, _) in enumerate(pairs)}
    out = [results[s] for s, _, _ in sorted(pairs, key=lambda x: order[x[0]])]
    return out

def make_fastp_step(
    out_root: str = "work/fastp",
    threads_per_job: int = 12,
    max_workers: int | None = None,
    backend: str = "process",
):
    """
    Input: FASTQ list [(srr, fq1, fq2), ...]
    Output: work/fastp/{SRR}_1.clean.fq.gz, {SRR}_2.clean.fq.gz
    Validation: cleaned FASTQ files exist and have size > 0.
    """
    def _cmd(fastq_triplets: Sequence[tuple[str, str, str]], logger=None):
        # Forward triplets directly to fastp_clean_parallel.
        # outdir is unused in triplet mode; provide a reasonable placeholder.
        os.makedirs(out_root, exist_ok=True)

        # Use the first fq1 directory as a placeholder outdir; default to "." when absent.
        if fastq_triplets:
            first_fq1_dir = os.path.dirname(fastq_triplets[0][1]) or "."
        else:
            first_fq1_dir = "."

        # Key detail: fastp_clean writes to {work_dir}/fastp/ while out_root = {work_dir}/fastp,
        # so choose work_dir = dirname(out_root).
        work_dir = os.path.dirname(out_root) or "."

        return fastp_clean_parallel(
            samples=list(fastq_triplets),           # Triplet mode [(srr, fq1, fq2), ...]
            outdir=first_fq1_dir,                   # Ignored in triplet mode.
            work_dir=work_dir,                      # Ensures outputs land in {work_dir}/fastp = out_root.
            fastp_threads=threads_per_job,
            max_workers=max_workers,
            retries=2,
            backend=backend
        )

    return {
        "name": "fastp",
        "command": _cmd,  # Accepts [(srr, fq1, fq2), ...].
        "outputs": [f"{out_root}" + "/{SRR}_1.clean.fq.gz",
                    f"{out_root}" + "/{SRR}_2.clean.fq.gz"],
        "validation": lambda fs: all(os.path.exists(f) and os.path.getsize(f) > 0 for f in fs),
        "takes": "FASTQ_PATHS",
        "yields": "CLEAN_FASTQ_PATHS"
    }
