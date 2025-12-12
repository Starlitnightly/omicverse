# star_step.py — batch STAR step (compatible with existing star_tools).
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple, List, Optional, Union
import os, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from .star_tools import star_align_auto     # Reuse the existing implementation.
from .tools_check import which_or_find, merged_env


def _bam_ok(p: Path) -> bool:
    """Treat BAM files larger than 1 MB as valid; having a .bai is preferable but optional."""
    try:
        return p.exists() and p.stat().st_size > 1_000_000
    except Exception:
        return False


def _normalize_bam(bam_path: Path, sample_dir: Path, srr: str) -> Path:
    """
    Normalize the BAM naming to <SRR>/Aligned.sortedByCoord.out.bam.
    - If star_align_auto produces run.Aligned.sortedByCoord.out.bam, move/rename it to the canonical name.
    - Move the corresponding .bai when present.
    Normalization steps:
      1) Ensure <SRR>/Aligned.sortedByCoord.out.bam exists (move the original product if needed).
      2) Provide an SRR-named handle <SRR>/<srr>.sorted.bam (prefer symlink; copy as fallback).
      3) Return the SRR handle for downstream featureCounts usage.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    target = sample_dir / "Aligned.sortedByCoord.out.bam"
    target_bai = target.with_suffix(".bam.bai")

    # Move into the standard location when the initial name differs.
    if bam_path.resolve() != target.resolve():
        # Capture the existing index name when present.
        src_bai = bam_path.with_suffix(".bam.bai")
        # Move the BAM.
        bam_path.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == 0:
            target.unlink()
        if not target.exists():
            shutil.move(str(bam_path), str(target))
        # Move the BAI.
        if src_bai.exists():
            if target_bai.exists():
                target_bai.unlink()
            shutil.move(str(src_bai), str(target_bai))

    # Ensure an index file is present.
    if not target_bai.exists():
        samtools = which_or_find("samtools")
        subprocess.run([samtools, "index", str(target)], check=True)

    # Create or refresh the SRR-named symlink.
    srr_bam = sample_dir / f"{srr}.sorted.bam"
    srr_bai = sample_dir / f"{srr}.sorted.bam.bai"
    try:
        if srr_bam.is_symlink() or srr_bam.exists():
            srr_bam.unlink()
        if srr_bai.is_symlink() or srr_bai.exists():
            srr_bai.unlink()
        # Relative links are more robust within the directory.
        srr_bam.symlink_to(target.name)
        srr_bai.symlink_to(target_bai.name)
    except OSError:
        # Copy the file when symlinks are unsupported.
        shutil.copy2(str(target), str(srr_bam))
        shutil.copy2(str(target_bai), str(srr_bai))

    return srr_bam


def _try_parse_index_dir(ret) -> Optional[Path]:
    """
    star_align_auto may return:
      - str(bam)
      - (bam,)
      - (bam, index_dir)
      - [bam, index_dir, ...]
    Attempt to parse index_dir; return None if it cannot be determined.
    """
    if isinstance(ret, (list, tuple)):
        if len(ret) >= 2:
            cand = Path(ret[1])
            if cand.exists() and cand.is_dir():
                return cand
    return None


def _align_one(
    srr: str,
    fq1: Union[str, Path],
    fq2: Union[str, Path, None],
    index_root: str,
    out_root: str,
    threads: int,
    gencode_release: str,
    sjdb_overhang: Optional[int],
    accession_for_species: Optional[str],
    memory_limit: str = "100G",  # BAM sorting memory limit
) -> Tuple[str, str, Optional[str]]:
    """
    Align a single sample; return (srr, bam_path, index_dir|None).
    """
    fq1_path = Path(fq1)
    fq2_path = Path(fq2) if fq2 else None
    sample_dir = Path(out_root) / srr
    sample_dir.mkdir(parents=True, exist_ok=True)
    # Normalize expected output paths (used for idempotency and downstream validation).
    expected_bam = sample_dir / "Aligned.sortedByCoord.out.bam"

    # Idempotent: skip when outputs already exist.
    if _bam_ok(expected_bam):
        print(f"[SKIP] STAR {srr}: {expected_bam}")
        # Ensure the SRR handle exists when missing.
        srr_bam = _normalize_bam(expected_bam, sample_dir, srr)
        return srr, str(srr_bam), None

    # Remain compatible with the original out_prefix="run.".
    out_prefix = sample_dir / "run."
    acc = accession_for_species or srr

    # Call the existing wrapper.
    ret = star_align_auto(
        accession=acc,
        fq1=str(fq1_path),
        fq2=str(fq2_path) if fq2_path else None,
        index_root=index_root,
        out_prefix=str(out_prefix),
        threads=threads,
        gencode_release=gencode_release,
        sjdb_overhang=sjdb_overhang,
        sample=srr,
        memory_limit=memory_limit,  # Add memory limit for BAM sorting
    )

    # Extract the BAM path from the return value.
    if isinstance(ret, (list, tuple)):
        bam_path = Path(ret[0])
    else:
        bam_path = Path(ret)

    if not bam_path.exists():
        raise FileNotFoundError(f"[STAR] BAM not found for {srr}: {bam_path}")

    # Normalize the BAM naming to <SRR>/Aligned.sortedByCoord.out.bam.
    bam_path = _normalize_bam(bam_path, sample_dir, srr)   # Pass srr through for naming consistency.
    idx_dir = _try_parse_index_dir(ret)
    return srr, str(bam_path), (str(idx_dir) if idx_dir else None)


def make_star_step(
    index_root: str = "index",
    out_root: str = "work/star",
    threads: int = 12,
    gencode_release: str = "v44",
    sjdb_overhang: int | None = 149,
    accession_for_species: str | None = None,  # Provide a shared accession when all samples belong to the same GSE.
    max_workers: int | None = None,            # Batch concurrency control; None keeps execution serial.
    memory_limit: str = "100G",                 # BAM sorting memory limit
):
    """
    Input: [(srr, fq1_clean, fq2_clean), ...]
    Output: work/star/{SRR}/Aligned.sortedByCoord.out.bam (one subdirectory per sample).
    Validation: BAM exists and is larger than 1 MB.
    """
    def _cmd(clean_fastqs: Sequence[Tuple[str, str | Path, str | Path | None]], logger=None) -> List[Tuple[str, str, Optional[str]]]:
        os.makedirs(out_root, exist_ok=True)

        # Default to serial execution when concurrency is unspecified (keeps logs orderly).
        if not max_workers or max_workers <= 1:
            products = []
            for srr, fq1, fq2 in clean_fastqs:
                rec = _align_one(
                    srr=srr, fq1=fq1, fq2=fq2,
                    index_root=index_root, out_root=out_root,
                    threads=threads, gencode_release=gencode_release,
                    sjdb_overhang=sjdb_overhang,
                    accession_for_species=accession_for_species,
                    memory_limit=memory_limit,
                )
                products.append(rec)  # (srr, bam, index_dir|None)
            return products

        # Concurrent mode (one task per sample).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        products: List[Tuple[str, str, Optional[str]]] = []
        errors: List[Tuple[str, str]] = []

        def _worker(item):
            srr, fq1, fq2 = item
            return _align_one(
                srr=srr, fq1=fq1, fq2=fq2,
                index_root=index_root, out_root=out_root,
                threads=threads, gencode_release=gencode_release,
                sjdb_overhang=sjdb_overhang,
                accession_for_species=accession_for_species,
                memory_limit=memory_limit,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_worker, it): it[0] for it in clean_fastqs}
            for fut in as_completed(futs):
                srr = futs[fut]
                try:
                    products.append(fut.result())
                except Exception as e:
                    msg = str(e)
                    errors.append((srr, msg))
                    if logger:
                        logger.error(f"[STAR] {srr} failed: {msg}")

        if errors:
            # Log errors but continue with successful samples.
            err_msg = "; ".join([f"{s}:{m}" for s, m in errors])
            if logger:
                logger.error(f"STAR failed for {len(errors)} samples: {err_msg}")
            # Continue with successful samples so downstream stages can proceed.
            if logger:
                logger.warning(f"Continuing with {len(products)} successful samples out of {len(clean_fastqs)}")
            # Raise an exception only if every sample fails.
            if len(products) == 0:
                raise RuntimeError(f"STAR failed for all samples: {err_msg}")

        # Preserve the original input ordering.
        order = {s: i for i, (s, _, _) in enumerate(clean_fastqs)}
        products.sort(key=lambda x: order[x[0]])
        return products

    return {
        "name": "star",
        "command": _cmd,  # Accepts [(srr, fq1_clean, fq2_clean), ...] and returns [(srr, bam, index_dir|None), ...]
        "outputs": [f"{out_root}" + "/{SRR}/Aligned.sortedByCoord.out.bam"],  # Matches the validation criteria.
        "validation": lambda fs: all(os.path.exists(f) and os.path.getsize(f) > 1_000_000 for f in fs),
        "takes": "CLEAN_FASTQ_PATHS",
        "yields": "BAM_PATHS"  # Returns triplets; downstream normalization remains in place.
    }
