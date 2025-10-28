# sra_prefetch.py Prefetch step factory
from __future__ import annotations
import os, time, subprocess, logging
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence
import shutil  
from typing import Dict, Any  
try:
    from tqdm.auto import tqdm 
except Exception:
    from tqdm import tqdm
    
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from .sra_tools import (get_sra_metadata, human_readable_speed, estimate_remaining_time, record_to_log, get_downloaded_file_size, run_prefetch_with_progress)
except ImportError:
    from sra_tools import (get_sra_metadata, human_readable_speed, estimate_remaining_time, record_to_log, get_downloaded_file_size, run_prefetch_with_progress)
# ---------- Utility helpers (aligned with your original implementation) ----------
def find_sra_file(srr_id: str, output_root: Path, timeout: int = 30) -> Path | None:
    """
    Locate the .sra/.sralite file for `srr_id` beneath `output_root`:
      1) Quickly check the root and immediate subdirectory.
      2) Recursively search the SRR directory.
      3) Poll until `timeout` seconds or until a non-empty file is found.
    """
    output_root = Path(output_root)
    srr_dir = output_root / srr_id

    def _ready(p: Path) -> bool:
        try:
            return p.exists() and p.stat().st_size > 0
        except Exception:
            return False

    # Quick direct checks first.
    quick = [
        output_root / f"{srr_id}.sra",
        output_root / f"{srr_id}.sralite",
        srr_dir / f"{srr_id}.sra",
        srr_dir / f"{srr_id}.sralite",
    ]
    for p in quick:
        if _ready(p):
            return p

    # Poll and recurse.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if srr_dir.exists():
            hits = list(srr_dir.rglob(f"{srr_id}.sr*"))  # Match .sra/.sralite.
            # Prioritize the newest file to avoid incomplete downloads.
            hits = [h for h in hits if _ready(h)]
            if hits:
                hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return hits[0]
        time.sleep(0.5)

    return None
def ensure_link_at(output_path: Path, real_file: Path, logger=None, prefer="symlink") -> Path:
    """
    Ensure `output_path` references `real_file`:
      - Prefer symbolic links; fall back to hardlink/copy when unsupported.
    Return output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        # Reuse the existing file.
        return output_path

    try:
        if prefer == "symlink":
            output_path.symlink_to(real_file)
        elif prefer == "hardlink":
            os.link(real_file, output_path)
        else:
            shutil.copy2(real_file, output_path)
        if logger:
            logger.info(f"[LINK] {output_path} -> {real_file}")
    except Exception as e:
        # Environments without symlink support: fall back to copying.
        if logger:
            logger.warning(f"[LINK] failed ({e}), fallback to copy")
        shutil.copy2(real_file, output_path)
    return output_path
    
def _make_local_logger(name: str, level=logging.INFO) -> logging.Logger:
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True) 
    logger = logging.getLogger(f"sraprefetch.{name}")
    if not logger.handlers:
        logger.setLevel(level)
        fh = logging.FileHandler(log_dir/f"{name}.log", mode="a")
        sh = logging.StreamHandler()
        fmt = logging.Formatter(f"[%(asctime)s] [{name}] %(levelname)s: %(message)s")
        for h in (fh, sh):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger

################################################
#                                              #
#   Compatibility bridge for Alignment class   #
#   and pipeline factory interfaces           #
#                                              #
################################################

# ---------- For Alignment Class ----------
def _monitor_file_size(path: Path, pbar: tqdm, stop_event: threading.Event, interval: float = 1.0):
    """Monitor file size externally and refresh the tqdm progress bar."""
    last = 0
    while not stop_event.is_set():
        try:
            size = path.stat().st_size if path.exists() else 0
            inc = max(0, size - last)
            if inc:
                pbar.update(inc)
                last = size
        except Exception:
            pass
        time.sleep(interval)
    # Final refresh at shutdown.
    try:
        size = path.stat().st_size if path.exists() else last
        if size > last:
            pbar.update(size - last)
    except Exception:
        pass
    pbar.close()
    
def prefetch_batch(
    srr_list: Sequence[str],
    out_root: str | Path,
    threads: int = 4, # Number of concurrent SRR downloads
    retries: int = 3,
    link_mode: str = "symlink",
    cache_root: str | Path = "sra_cache",
    prefetch_config: Optional[Dict[str, Any]] = None,  # Base prefetch configuration
):
    """
    Adapter to integrate with Alignment.prefetch().
    For each SRR:
      1) call run_prefetch_with_progress(srr, logger, retries, output_root="sra_cache", prefetch_config=prefetch_config)
      2) locate the real .sra in the cache
      3) place a link/copy at <out_root>/<SRR>/<SRR>.sra
      Parallel SRA downloads: prefetch each SRR in srr_list concurrently.
      `threads` controls the number of concurrent download jobs.
      After download, locate the .sra under cache_root and link/copy to out_root/<SRR>/<SRR>.sra.

    """
    out_root = Path(out_root)
    cache_root = out_root 
    out_root.mkdir(parents=True, exist_ok=True)
    
    logger = _make_local_logger("prefetch")
    # Single download task.
    def _worker(pos: int, srr: str) -> tuple[str, Path]:
        
        # Target directory (mirrors the original landing location).
        srr_dir = cache_root 
        #srr_dir.mkdir(parents=True, exist_ok=True)
        #sra_file = srr_dir / f"{srr}.sra"

        # Download using the base prefetch configuration.
        ok = run_prefetch_with_progress(
            srr_id=srr, logger=logger, retries=retries, output_root=str(cache_root),
            prefetch_config=prefetch_config  # Use the base configuration.
        )

        # After the command returns successfully:
        real = find_sra_file(srr_id=srr, output_root=cache_root, timeout=30)
        if not real:
            logger.error(f"Prefetch completed but cannot locate .sra/.sralite for {srr} under {cache_root}")
            # Optional: emit a debug tree of the cache folder.
            try:
                for p in (cache_root / srr).rglob("*"):
                    logger.debug(f"[tree] {p}")
            except Exception:
                pass
            raise RuntimeError(f"Prefetch completed but .sra/.sralite not found for {srr}")

        # Preserve the original suffix to avoid mislabeling .sralite as .sra.
        dest = out_root / f"{srr}{Path(real).suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        ensure_link_at( Path(real), dest, prefer=link_mode, logger=logger)
        return srr, dest

    results_map: dict[str, Path] = {}
    failures: list[tuple[str, Exception]] = []

    # Execute multiple downloads concurrently.
    with ThreadPoolExecutor(max_workers=int(threads)) as pool:
        fut_map = {pool.submit(_worker, pos,srr): srr for pos, srr in enumerate(srr_list)}
        for fut in as_completed(fut_map):
            srr = fut_map[fut]
            try:
                srr_id, dest_path = fut.result()
                results_map[srr_id] = dest_path
                logger.info(f"[prefetch] OK: {srr_id} -> {dest_path}")
            except Exception as e:
                failures.append((srr, e))
                logger.error(f"[prefetch] FAIL: {srr}: {e}")

    if failures:
        failed = ", ".join([f"{srr}({repr(err)})" for srr, err in failures])
        raise RuntimeError(f"Prefetch failed for {len(failures)} samples: {failed}")
    # Preserve the original input order.
    return [results_map[s] for s in srr_list if s in results_map]

# ---------- Step factory: invoke from orchestration modules when needed ----------
def make_prefetch_step(output_pattern: str = "sra_cache/{SRR}/{SRR}.sra", validation=None):
    def _safe_valid(fs: list[str]) -> bool:
        # fs contains a single expected path; fall back to find_sra_file as a safeguard.
        try:
            f = Path(fs[0])
            if f.exists() and f.stat().st_size > 1_000_000:
                return True
            # Use the locator helper to confirm the cached file.
            real = find_sra_file(f.stem, f.parent.parent)  # stem=SRRxxxxxx, parent.parent=output_root
            return bool(real and real.exists() and real.stat().st_size > 1_000_000)
        except Exception:
            return False

    return {
        "name": "prefetch",
        "command": run_prefetch_with_progress,
        "outputs": [output_pattern],
        "validation": validation or _safe_valid,
    }
