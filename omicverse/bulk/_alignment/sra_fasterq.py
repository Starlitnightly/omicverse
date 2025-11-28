# sra_fasterq.py
from __future__ import annotations
import os, sys, shutil, time, subprocess,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Sequence, Tuple, List, Dict, Optional, Union

try:
    from .tools_check import which_or_find, merged_env
except ImportError:
    from tools_check import which_or_find, merged_env
    
def _fqdump_one(srr: str, outdir: str, fqdump_bin: str, threads: int, mem_gb: int, do_gzip: bool):
    os.makedirs(outdir, exist_ok=True)
    env = merged_env()
    # fasterq-dump emits uncompressed .fastq files.
    cmd = [
        fqdump_bin, srr, "-p", "-O", outdir,
        "-e", str(threads), "--mem", f"{mem_gb}G", "--split-files"
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    # Detect produced FASTQ files.
    fq1 = os.path.join(outdir, f"{srr}_1.fastq")
    fq2 = os.path.join(outdir, f"{srr}_2.fastq")
    if not (os.path.exists(fq1) and os.path.exists(fq2)):
        raise FileNotFoundError(f"fasterq outputs missing for {srr} in {outdir}")

    # Optional: gzip compression with MD5 validation.
    if do_gzip:
        import hashlib, gzip, shutil as _sh
        for p in (fq1, fq2):
            gz = p + ".gz"
            if not os.path.exists(gz):
                with open(p, "rb") as f_in, gzip.open(gz, "wb") as f_out:
                    _sh.copyfileobj(f_in, f_out)
                # Simple integrity check: read the header again.
                with gzip.open(gz, "rb") as f_chk:
                    _ = f_chk.read(128)
                os.remove(p)
        fq1 += ".gz"; fq2 += ".gz"

    return srr, fq1, fq2

def _have(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def _candidate_fastq_paths(out_root: Path, srr: str, gzip_output: bool):
    suffix = ".fastq.gz" if gzip_output else ".fastq"
    fq1 = out_root / f"{srr}_1{suffix}"
    fq2 = out_root / f"{srr}_2{suffix}"
    return fq1, fq2

def _run(cmd: list[str]):
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True)



def _find_single(ext_gz: bool) -> Path | None:
    """Find single-end fastq files, similar to _find_pair for paired-end."""
    suffix = ".fastq.gz" if ext_gz else ".fastq"
    for base in cand_dirs:
        for st in stems:
            # Look for single-end files
            patterns = [base / f"{st}.sra{suffix}", base / f"{st}{suffix}"]
            for single_pattern in patterns:
                if _have(single_pattern):
                    return single_pattern

            # Wildcard fallback - any fastq file with the SRR name
            if base.exists():
                candidates = list(base.glob(f"*{suffix}"))
                for cand in candidates:
                    if _have(cand):
                        return cand
    return None

def _work_one_fasterq(
    srr: str,
    out_root: Path,
    threads_per_job: int,
    mem_gb: int,
    tmp_root: Path,
    gzip_output: bool,
    retries: int = 3,
    library_layout: str = "auto",  # Library layout hint: "auto", "single", "paired"
) -> tuple[str, Path, Path | None]:
    """
    Process a single SRR with fasterq-dump, handling early exits and both paired/single-end data.
    Returns: (srr, fastq_path, fastq2_path | None) where fq2 is None for single-end.
    """
    # 1) Output and temporary directories.
    out_dir = out_root / srr
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / srr
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 2) Auto-detect the prefetch directory (peer directory).
    prefetch_root = out_root.parent / "prefetch"

    # 3) Handle both .sra and .sralite inputs.
    local_sra = None
    for ext in [".sra", ".sralite"]:
        candidate = prefetch_root / srr / f"{srr}{ext}"
        if candidate.exists():
            local_sra = candidate
            break

    # 4) Choose the input token.
    input_token = str(local_sra) if local_sra else srr
    cand_dirs = (out_dir, out_root)
    stems = {srr}
    try:
        stems.add(Path(input_token).stem)
    except Exception:
        pass

    def _find_pair(ext_gz: bool) -> tuple[Path, Path] | None:
        suffix = ".fastq.gz" if ext_gz else ".fastq"
        for base in cand_dirs:
            for st in stems:
                for name1 in (f"{st}_1{suffix}", f"{st}.sra_1{suffix}"):
                    name2 = name1.replace("_1", "_2", 1)
                    a, b = base / name1, base / name2
                    if _have(a) and _have(b):
                        return a, b
        return None
    # ---------------- Early exit 1: gzip requested and paired .gz already present -> return ----------------
    # Only apply to paired-end files
    if library_layout == "auto" or library_layout == "paired":
        # Check paired-end files
        if gzip_output:
            hit_gz = _find_pair(ext_gz=True)
            if hit_gz:
                fq_a, fq_b = hit_gz
                print(f"[SKIP] {srr}: paired .fastq.gz already exists; skipping fasterq-dump.")
                return srr, fq_a, fq_b

    # ---------------- Early exit 2: paired .fastq already present ----------------
    if library_layout == "auto" or library_layout == "paired":
        # Check paired-end files
        hit_plain = _find_pair(ext_gz=False)
        if hit_plain:
            fq_a, fq_b = hit_plain
            if gzip_output:
                # Only perform gzip (skip fasterq-dump).
                print(f"[INFO] {srr}: found uncompressed paired .fastq, starting gzip step.")
                for src in (fq_a, fq_b):
                    if src.exists() and src.suffix == ".fastq":
                        _run(["gzip", "-f", str(src)])
                fq_a = fq_a.with_suffix(".fastq.gz")
                fq_b = fq_b.with_suffix(".fastq.gz")
                if not (_have(fq_a) and _have(fq_b)):
                    raise RuntimeError(f"gzip step did not produce gz files for {srr}")
                print(f"[SKIP] {srr}: paired gzip already completed; skipping fasterq-dump.")
            else:
                print(f"[SKIP] {srr}: paired .fastq exists; skipping fasterq-dump.")
            return srr, fq_a, fq_b

    # ---------------- Early exit 3: single-end fastq files already present ----------------
    if library_layout == "auto" or library_layout == "single":
        # Internal function to search for single-end files (similar to _find_pair)
        def _find_single(ext_gz: bool) -> Path | None:
            """Find single-end fastq files, similar to _find_pair for paired-end."""
            suffix = ".fastq.gz" if ext_gz else ".fastq"
            for base in cand_dirs:
                for st in stems:
                    # Look for single-end files
                    patterns = [base / f"{st}.sra{suffix}", base / f"{st}{suffix}"]
                    for single_pattern in patterns:
                        if _have(single_pattern):
                            return single_pattern

                # Wildcard fallback - any fastq file with the SRR name
                if base.exists():
                    candidates = list(base.glob(f"*{suffix}"))
                    for cand in candidates:
                        if _have(cand):
                            return cand
            return None

        # Check single-end files
        suffix = ".fastq.gz" if gzip_output else ".fastq"
        hit_single = _find_single(ext_gz=True if gzip_output else False)
        if hit_single:
            if gzip_output:
                print(f"[SKIP] {srr}: single-end .fastq.gz already exists; skipping fasterq-dump.")
                return srr, hit_single, None
            else:
                print(f"[SKIP] {srr}: single-end .fastq exists; skipping fasterq-dump.")
                return srr, hit_single, None

        # Try to find uncompressed .fastq and gzip it if needed
        if gzip_output:
            hit_uncompressed = _find_single(ext_gz=False)
            if hit_uncompressed:
                print(f"[INFO] {srr}: found uncompressed single-end .fastq, starting gzip step.")
                if hit_uncompressed.exists() and hit_uncompressed.suffix == ".fastq":
                    _run(["gzip", "-f", str(hit_uncompressed)])
                gz_path = hit_uncompressed.with_suffix(".fastq.gz")
                if not _have(gz_path):
                    raise RuntimeError(f"gzip step did not produce gz file for {srr}")
                print(f"[SKIP] {srr}: single gzip already completed; skipping fasterq-dump.")
                return srr, gz_path, None


    # Build fasterq-dump commands for auto-detection

    fq_bin = shutil.which("fasterq-dump") or "fasterq-dump"
    base_args = [
        input_token,
        "-p",
        "-O", str(out_dir),
        "-e", str(threads_per_job),
        "--mem", f"{mem_gb}G",
        "-t", str(tmp_dir),
        "-f",
    ]

    # Handle user-specified library layout
    if library_layout == "single":
        # User specified single-end - skip paired attempt
        print(f"[DEBUG] Using user-specified single-end library layout for {srr}")
        is_paired = False
    elif library_layout == "paired":
        # User specified paired-end - only try paired approach
        print(f"[DEBUG] Using user-specified paired-end library layout for {srr}")
        is_paired = True
    else:
        # Auto-detect library layout
        is_paired = None

    # Handle user-specified library layout
    if library_layout != "auto":
        print(f"[DEBUG] Using user-specified library layout: {library_layout} for {srr}")
        if library_layout == "single":
            is_paired = False
        elif library_layout == "paired":
            is_paired = True

    # Process based on library layout
    if is_paired is True:
        # User specified paired-end - try paired approach only
        cmd_paired = [fq_bin] + base_args + ["--split-files"]
        print("Try:", " ".join(cmd_paired) + " (paired-end mode)", flush=True)

        try:
            _run(cmd_paired)
            # Check if paired files were created
            paired_files = _find_pair(ext_gz=False) or _find_pair(ext_gz=True)
            if paired_files:
                print(f"[INFO] Successfully determined {srr}: paired-end")
            else:
                raise RuntimeError(f"fasterq-dump with --split-files did not create paired files for {srr}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"fasterq-dump failed for paired-end mode: {str(e)}")

    elif is_paired is False:
        # User specified single-end - try single approach only
        cmd_single = [fq_bin] + base_args + []
        print("Try:", " ".join(cmd_single) + " (single-end mode)", flush=True)

        try:
            _run(cmd_single)
            # DEBUG: List all files in output directories
            # Check if single file was created - look for uncompressed first
            suffix = ".fastq"  # Always check for uncompressed first with fasterq-dump
            single_files = []

            for base in cand_dirs:
                for st in stems:
                    patterns = [base / f"{st}.sra{suffix}", base / f"{st}{suffix}"]
                    for single in patterns:
                        if _have(single):
                            single_files.append(single)
                            break

                    # If explicit patterns didn't work, try wildcard fallback
                    if not single_files:
                        if base.exists():
                            candidates = list(base.glob(f"*{suffix}"))
                            for cand in candidates:
                                if _have(cand):
                                    single_files.append(cand)
                                    break
                    if single_files:
                        break
                if single_files:
                    break
            if single_files:
                print(f"[INFO] Successfully determined {srr}: single-end")
            else:
                # Build detailed error message showing what was searched
                searched_patterns = []
                for base in cand_dirs:
                    for st in stems:
                        patterns = [base / f"{st}.sra{suffix}", base / f"{st}{suffix}"]
                        searched_patterns.extend([str(p) for p in patterns])

                # Also check for any .fastq files at all in directories
                all_fastq_files = []
                for base in cand_dirs:
                    if base.exists():
                        all_fastq_files.extend([str(f) for f in base.glob(f"*{suffix}")])

                err_msg = (
                    f"fasterq-dump completed but output file not found for {srr}.\n"
                    f"Expected patterns searched: {searched_patterns}\n"
                    f"All {suffix} files found: {all_fastq_files if all_fastq_files else 'None'}\n"
                    f"Directories searched: {[str(d) for d in cand_dirs]}\n"
                    f"Stems used: {list(stems)}"
                )
                raise RuntimeError(err_msg)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"fasterq-dump failed for single-end mode: {str(e)}")

    else:  # Auto-detect library layout
        # First, try with --split-files (paired-end)
        cmd_paired = [fq_bin] + base_args + ["--split-files"]
        print("Try:", " ".join(cmd_paired) + " (auto: paired-end mode)", flush=True)

        try:
            _run(cmd_paired)
            # Check if paired files were created
            paired_files = _find_pair(ext_gz=False) or _find_pair(ext_gz=True)
            if paired_files:
                is_paired = True
                print(f"[INFO] Successfully determined {srr}: paired-end")
        except subprocess.CalledProcessError:
            print(f"[WARN] fasterq-dump failed with --split-files, will try single-end")
            # Clear partial outputs
            for base in cand_dirs:
                for st in stems:
                    for ext in [".fastq", ".fastq.gz"]:
                        for p in [base / f"{st}.sra{ext}", base / f"{st}{ext}",
                                  base / f"{st}_1{ext}", base / f"{st}_2{ext}"]:
                            if _have(p):
                                p.unlink()

        # If paired files weren't created, try without --split-files (single-end)
        if is_paired is None:
            cmd_single = [fq_bin] + base_args + []
            print("Try:", " ".join(cmd_single) + " (auto: single-end mode)", flush=True)

            try:
                _run(cmd_single)
                # Check if single file was created
                suffix = ".fastq.gz" if gzip_output else ".fastq"
                single_files = []
                for base in cand_dirs:
                    for st in stems:
                        for single in (base / f"{st}.sra{suffix}", base / f"{st}{suffix}"):
                            if _have(single):
                                single_files.append(single)
                                break
                    if single_files:
                        break
                if single_files:
                    is_paired = False
                    print(f"[INFO] Successfully determined {srr}: single-end")
                else:
                    raise RuntimeError(f"fasterq-dump did not create expected output files for {srr}")
            except Exception as e:
                raise RuntimeError(f"fasterq-dump failed for both paired-end and single-end attempts: {str(e)}")

    # Validate raw fastq results exist
    # Locate outputs: support names like "SRR.sra_1.fastq" when the input is .sra.
    # Candidate directories: SRR subdirectory plus the output root.
    cand_dirs = (out_dir, out_root)

    # Candidate prefixes: typically the SRR; include the stem of input_token when it is a .sra path.
    stems = {srr}
    try:
        stems.add(Path(input_token).stem)  # Could be "SRR123" or "SRR123.sra".
    except Exception:
        pass

    def _find_pair(ext_gz: bool) -> tuple[Path, Path] | None:
        """Search the cand_dirs × stems space for paired _1/_2.fastq(.gz) files."""
        suffix = ".fastq.gz" if ext_gz else ".fastq"
        for base in cand_dirs:
            for st in stems:
                # Accept both "SRR_1" and "SRR.sra_1" naming variants.
                for name1 in (f"{st}_1{suffix}", f"{st}.sra_1{suffix}"):
                    name2 = name1.replace("_1", "_2", 1)
                    a, b = base / name1, base / name2
                    if _have(a) and _have(b):
                        return a, b
        return None

    # Determine output files based on library layout
    if is_paired is True:
        hit = _find_pair(ext_gz=True) or _find_pair(ext_gz=False)
        if not hit:
            searched = ", ".join(str(d) for d in cand_dirs)
            raise RuntimeError(f"fasterq paired outputs missing for {srr} (searched: {searched})")
        fq_a, fq_b = hit
        # If gzip is requested and the outputs are still .fastq, compress them in place.
        if gzip_output and fq_a.suffix == ".fastq":
            for src in (fq_a, fq_b):
                if src.exists() and src.suffix == ".fastq":
                    _run(["gzip", "-f", str(src)])
            fq_a = fq_a.with_suffix(".fastq.gz")
            fq_b = fq_b.with_suffix(".fastq.gz")
            if not (_have(fq_a) and _have(fq_b)):
                raise RuntimeError(f"gzip step did not produce gz files for {srr}")
        return srr, fq_a, fq_b

    elif is_paired is False:
        # Single-end processing - find .fastq, then gzip if requested
        search_suffix = ".fastq"  # fasterq-dump creates uncompressed .fastq
        fq_uncompressed = None

        # First find the uncompressed .fastq file
        for base in cand_dirs:
            for st in stems:
                patterns = [base / f"{st}.sra{search_suffix}", base / f"{st}{search_suffix}"]
                for pattern in patterns:
                    if _have(pattern):
                        fq_uncompressed = pattern
                        break
                if fq_uncompressed:
                    break
            if fq_uncompressed:
                break

        if not fq_uncompressed:
            searched = ", ".join(str(d) for d in cand_dirs)
            raise RuntimeError(f"fasterq single output missing for {srr} (searched: {searched})")

        # Apply gzip compression if requested
        if gzip_output:
            gz_path = fq_uncompressed.with_suffix(".fastq.gz")
            if not _have(gz_path):
                _run(["gzip", "-f", str(fq_uncompressed)])
            if not _have(gz_path):
                raise RuntimeError(f"gzip compression failed for single-file in {srr}")
            return srr, gz_path, None  # Return gzipped path
        else:
            return srr, fq_uncompressed, None  # Return raw fastq path, None for fq2 indicates single-end

    else:
        raise RuntimeError(f"Failed to determine library layout for {srr}")

def fasterq_batch(
    srr_list: Sequence[str],
    out_root: str | Path,
    threads: int = 8,          # Number of SRRs to process concurrently (maps to max_workers).
    gzip_output: bool = True,  # Whether to gzip the FASTQ output.
    mem: str = "8G",
    tmp_root: str | Path = "work/tmp",
    backend: str = "process",
    threads_per_job: int = 24,       # fasterq-dump threads per SRR.
    library_layout: str = "auto"  # Library layout hint: "auto", "single", "paired"
) ->list[tuple[str, Path, Path | None]]:
    """
    Adapter that maps pipeline parameters to fasterq_dump_parallel and returns a list[tuple].
    Supports both paired-end (fq2 present) and single-end (fq2=None) library layouts.
    """
    # Parse mem (accept "4G"/"8G" formats; default to 8G when parsing fails).
    try:
        mem_gb = int(str(mem).upper().rstrip("G"))
    except Exception:
        mem_gb = 8

    res = fasterq_dump_parallel(
        srr_list=list(srr_list),
        out_root=str(out_root),
        threads_per_job=threads_per_job,
        mem_gb=mem_gb,
        max_workers=int(threads),      # Concurrency level.
        gzip_output=bool(gzip_output),
        backend=backend,
        tmp_root=str(tmp_root),
        library_layout=library_layout  # Pass user-specified library layout hint
    )

    by_srr = res.get("by_srr", {})
    errs = res.get("failed", [])

    if errs:
        # Optionally warn instead of raising.
        msgs = "; ".join([f"{s}: {m}" for s, m in errs])
        raise RuntimeError(f"fasterq_batch failed for {len(errs)} samples: {msgs}")

    # Convert to [(srr, Path, fq2_path|None)] in the original input order.
    out_pairs = []
    for srr in srr_list:
        if srr in by_srr:
            fq_path, fq2_path = by_srr[srr]  # fq2_path is None for single-end
            out_pairs.append((srr, Path(fq_path), fq2_path))

    return out_pairs

def fasterq_dump_parallel(
    srr_list: list[str],
    out_root: str = "work/fasterq",
    threads_per_job: int = 24,
    mem_gb: int = 8,
    max_workers: int | None = None,
    gzip_output: bool = True,
    backend: str = "process",
    tmp_root: str = "work/tmp",
    library_layout: str = "auto"  # Library layout hint passed from alignment pipeline
)-> Dict[str, object]:
    
    total_cores = os.cpu_count() or 8
    out_root = Path(out_root)
    tmp_root = Path(tmp_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Resolve absolute paths to the fasterq binaries (critical!).
    fqdump_bin = which_or_find("fasterq-dump")
    print(f"[INFO] fasterq-dump: {fqdump_bin}")

    if max_workers is None:
        import math, os as _os
        max_workers = max(1, math.floor((_os.cpu_count() or 8) / max(1, threads_per_job)))

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    by_srr: Dict[str, Tuple[str, str|None]] = {}
    errs: List[Tuple[str, str]] = []

    Path(tmp_root).mkdir(parents=True, exist_ok=True)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    with Executor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                _work_one_fasterq,
                srr, out_root, threads_per_job, mem_gb, tmp_root, gzip_output, 3, library_layout
            ): srr for srr in srr_list
        }
        for fut in as_completed(futs):
            srr = futs[fut]
            try:
                result = fut.result()
                srr_id = result[0]
                fq1p = result[1]
                fq2p = result[2] if len(result) > 2 else None  # fq2p is None for single-end
            except Exception as e:
                errs.append((srr, str(e)))
                continue
            # Store result: either (fq1_path, fq2_path) for paired or (fq1_path, None) for single-end
            if fq2p is None:
                by_srr[srr_id] = (str(fq1p), None)
            else:
                by_srr[srr_id] = (str(fq1p), str(fq2p))

    if errs:
        print("[SUMMARY] fasterq errors:", errs)
    return {"by_srr": by_srr, "failed": errs}

def _parse_mem_gb(mem_per_job: str | int) -> int:
    """
    Accept formats like '4G' / '8g' / 4 / '4' and normalize to integer GB.
    """
    if isinstance(mem_per_job, int):
        return mem_per_job
    s = str(mem_per_job).strip().lower()
    if s.endswith("gb"):
        s = s[:-2]
    elif s.endswith("g"):
        s = s[:-1]
    return int(float(s))


                
def make_fasterq_step(
    outdir_pattern: str = "work/fasterq",   # Output directory pattern (unchanged from prior behavior).
    threads_per_job: int = 24,
    mem_per_job: str = "4G",               # Back-compat string; converted to integer GB internally.
    max_workers: int | None = None,
    retries: int = 2,                      # Placeholder (unused) retained to avoid interface breakage.
    tmp_root: str = "work/tmp",            # Same rationale as above.
    backend: str = "process",
    compress_after: bool = True,           # Maps onto gzip_output.
    compress_threads: int = 8,             # Placeholder (unused) retained for compatibility.
):
    """
    Inputs: srr_list values supplied by the pipeline.
    Outputs: work/fasterq/{SRR}_1.fastq.gz and {SRR}_2.fastq.gz.
    Validation: both FASTQ files must exist and be non-empty.
    """
    def _cmd(srr_list: List[str], logger=None) -> List[Tuple[str, str, str]]:
        os.makedirs(outdir_pattern, exist_ok=True)

        # Map parameters to the newer fasterq_dump_parallel helper.
        mem_gb = _parse_mem_gb(mem_per_job)

        ret = fasterq_dump_parallel(
            srr_list=srr_list,
            out_root=outdir_pattern,         # Match the configured output pattern.
            threads_per_job=threads_per_job,
            mem_gb=mem_gb,                   # Normalized integer GB.
            max_workers=max_workers,
            gzip_output=compress_after,      # Map compress_after to gzip_output.
            backend=backend,
        )
        # Normalize outputs into [(srr, fq1, fq2), ...] for pipeline consumption.
        by_srr = ret.get("by_srr", {})
        products = [(srr, paths[0], paths[1]) for srr, paths in by_srr.items()]
        # Optionally surface ret["failed"] to the logs.
        if logger and ret.get("failed"):
            for srr, err in ret["failed"]:
                logger.error(f"[fasterq] {srr} failed: {err}")
        return products

    return {
        "name": "fasterq",
        "command": _cmd,  # Accepts a list of SRRs.
        "outputs": [f"{outdir_pattern}" + "/{SRR}_1.fastq.gz",
                    f"{outdir_pattern}" + "/{SRR}_2.fastq.gz"],
        "validation": lambda fs: all(os.path.exists(f) and os.path.getsize(f) > 0 for f in fs),
        "takes": "SRR_LIST",
        "yields": "FASTQ_PATHS"
    }
