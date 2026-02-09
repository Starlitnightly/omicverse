"""SRA prefetch wrapper with validation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .._registry import register_function
from ._cli_utils import (
    build_env,
    ensure_dir,
    ensure_link,
    find_sra_file,
    listify,
    resolve_executable,
    resolve_jobs,
    run_cmd,
    run_in_threads,
)


@register_function(
    aliases=["prefetch", "sra_prefetch", "sra-download"],
    category="alignment",
    description="Download SRA accessions with prefetch and verify integrity via vdb-validate.",
    examples=[
        "ov.alignment.prefetch('SRR1234567', output_dir='prefetch')",
        "ov.alignment.prefetch(['SRR1', 'SRR2'], output_dir='prefetch', jobs=4)",
    ],
    related=["alignment.fqdump", "alignment.fastp", "alignment.STAR", "alignment.featureCount"],
)
def prefetch(
    sra_ids: Union[str, Sequence[str]],
    output_dir: str = "prefetch",
    threads: int = 4,
    jobs: Optional[int] = None,
    max_workers: Optional[int] = None,
    retries: int = 2,
    validate: bool = True,
    transport: Optional[str] = None,
    location: Optional[str] = None,
    prefetch_path: Optional[str] = None,
    vdb_validate_path: Optional[str] = None,
    link_mode: str = "symlink",
    auto_install: bool = True,
    progress_minutes: int = 1,
    force: Optional[str] = None,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """
    Prefetch SRA accessions with validation.

    Parameters
    ----------
    sra_ids
        SRR accession (str) or list of accessions.
    output_dir
        Output directory for downloaded .sra/.sralite files.
    threads
        Default concurrency when jobs is not provided (kept for compatibility).
    jobs
        Number of concurrent downloads (preferred).
    max_workers
        Legacy alias for jobs.
    retries
        Retries per accession.
    validate
        Run vdb-validate on the downloaded file.
    transport
        Optional prefetch --transport value (e.g. 'https').
    location
        Optional prefetch --location value (e.g. 'ncbi', 'ena').
    prefetch_path
        Explicit path to prefetch executable.
    vdb_validate_path
        Explicit path to vdb-validate executable.
    link_mode
        symlink, hardlink, or copy (fallback).
    auto_install
        Install missing tools automatically when possible.
    progress_minutes
        Prefetch progress interval in minutes (0 disables progress).
    force
        Optional prefetch force mode: "no", "yes", or "all".
    """
    srrs = listify(sra_ids)
    ensure_dir(output_dir)

    if jobs is None and max_workers is None:
        jobs = threads
    worker_count = resolve_jobs(len(srrs), jobs, max_workers)

    prefetch_bin = resolve_executable("prefetch", prefetch_path, auto_install=auto_install)
    vdb_bin = resolve_executable("vdb-validate", vdb_validate_path, auto_install=auto_install) if validate else None

    extra_paths = [str(Path(prefetch_bin).parent)]
    if vdb_bin:
        extra_paths.append(str(Path(vdb_bin).parent))
    env = build_env(extra_paths=extra_paths)

    def _run_one(srr: str) -> Dict[str, str]:
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                cmd = [
                    prefetch_bin,
                    srr,
                    "-O",
                    str(output_dir),
                    "-p",
                    str(progress_minutes),
                ]
                if transport:
                    cmd.extend(["--transport", transport])
                if location:
                    cmd.extend(["--location", location])
                if force:
                    cmd.extend(["-f", force])

                run_cmd(cmd, env=env)
                sra_path = find_sra_file(srr, output_dir)
                if not sra_path:
                    raise FileNotFoundError(f"SRA file not found for {srr} under {output_dir}")

                if validate and vdb_bin:
                    run_cmd([vdb_bin, str(sra_path)], env=env)

                dest_dir = Path(output_dir) / srr
                dest_path = dest_dir / f"{srr}{sra_path.suffix}"
                dest_path = ensure_link(dest_path, sra_path, prefer=link_mode)

                return {
                    "srr": srr,
                    "path": str(dest_path),
                    "validated": str(bool(validate)),
                }
            except Exception as exc:
                last_err = exc
                time.sleep(min(5 * attempt, 20))
        raise RuntimeError(f"prefetch failed for {srr}: {last_err}")

    results = run_in_threads(srrs, _run_one, worker_count)
    if isinstance(sra_ids, str):
        return results[0]
    return results
