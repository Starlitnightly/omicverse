# qc_fastp.py fastp step factory
from __future__ import annotations

try:
    from .qc_tools import fastp_clean_parallel
except ImportError:
    from qc_tools import fastp_clean_parallel

import os, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, Tuple, List, Dict

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
    fq2: Path,
    out_root: Path,
    threads: int = 8,
) -> Tuple[str, Path, Path, Path, Path]:
    """
    Run fastp for a single paired-end sample.
    Returns (srr, clean1, clean2, json, html).
    """
    env = merged_env()
    fastp_bin = which_or_find("fastp")  # Resolve the executable path.
    out_dir = _ensure_dir(out_root / srr)

    # Compose output filenames (respect the input suffix behaviour).
    oext = _ext(fq1)
    clean1 = out_dir / f"{srr}_clean_1{oext.replace('.fastq', '')}"
    clean2 = out_dir / f"{srr}_clean_2{oext.replace('.fastq', '')}"
    json = out_dir / f"{srr}.fastp.json"
    html = out_dir / f"{srr}.fastp.html"

    # Skip when outputs already exist (override if you prefer forced rewrites).
    if clean1.exists() and clean2.exists() and json.exists() and html.exists():
        return srr, clean1, clean2, json, html

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
        "--thread", str(threads),   # Some versions accept --thread in addition to -w.
        "--overrepresentation_analysis",
    ]

    # fastp compresses automatically when the suffix is .gz.
    # Optional stricter filtering parameters (quality, unqualified %, minimum length):
    # cmd += ["-q", "20", "-u", "30", "-l", "30"]

    _run(cmd, env=env)

    # Basic validation.
    if not (clean1.exists() and clean1.stat().st_size > 0 and clean2.exists() and clean2.stat().st_size > 0):
        raise RuntimeError(f"fastp outputs missing/empty for {srr} in {out_dir}")

    return srr, clean1, clean2, json, html


def fastp_batch(
    pairs: Sequence[Tuple[str, Path, Path]],  # [(srr, fq1, fq2), ...]
    out_root: str | Path,
    threads: int = 12,          # fastp threads per sample.
    max_workers: int | None = None,  # Number of samples processed concurrently; None lets us auto-select.
) -> List[Tuple[str, Path, Path, Path, Path]]:
    """
    Execute fastp concurrently for multiple samples.
    Returns [(srr, clean1, clean2, json, html), ...] in the original ordering.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        # Default concurrency: min(threads, cpu_count/2); adjust to taste.
        import os, math
        max_workers = max(1, min(threads, (os.cpu_count() or 8) // 2))

    results: Dict[str, Tuple[str, Path, Path, Path, Path]] = {}
    errors: List[Tuple[str, str]] = []

    def _worker(item: Tuple[str, Path, Path]):
        srr, fq1, fq2 = item
        return _fastp_one(srr, Path(fq1), Path(fq2), out_root=out_root, threads=threads)

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
