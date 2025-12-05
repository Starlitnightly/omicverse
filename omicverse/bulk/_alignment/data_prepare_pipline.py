# Author: Zhi Luo
# pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Union, List
import pandas as pd
from .sra_prefetch import make_prefetch_step, run_prefetch_with_progress, _make_local_logger
from .sra_fasterq import make_fasterq_step
from .qc_fastp import make_fastp_step
from .star_step import make_star_step
from .count_step import make_featurecounts_step
from . import tools_check
import os

# 1) Declaratively compose steps (tune parameters per project needs).
tools_check.check()
STEPS = [
    make_prefetch_step(),
    make_fasterq_step(outdir_pattern="work/fasterq", threads_per_job=12, compress_after=True),
    make_fastp_step(out_root="work/fastp", threads_per_job=12),
    make_star_step(index_root="index", out_root="work/star", threads=12, gencode_release="v44", sjdb_overhang=149),  
    make_featurecounts_step(out_root="work/counts", simple=True, gtf=None, by="auto", threads=8)
]

def _render_paths(patterns, **kwargs):
    return [p.format(**kwargs) for p in patterns]
def _safe_call_validation(validation_fn, outs, logger):
    try:
        return bool(validation_fn(outs))
    except FileNotFoundError:
        # ✅ Even if the validation function misuses getsize, it will not crash here.
        logger.warning(f"[VALIDATION] missing file(s): {outs}")
        return False
    except Exception as e:
        logger.warning(f"[VALIDATION] raised {type(e).__name__}: {e} | outs={outs}")
        return False
def _normalize_srr_list(
    srrs_or_csv: Union[str, Path, Iterable[str], pd.DataFrame]
) -> List[str]:
    """
    Accept SRR inputs via list / CSV path / DataFrame.
    - CSV: prefer a 'Run' column (case-insensitive), then 'run_accession'.
    - Automatically deduplicate while preserving order.
    """
    if isinstance(srrs_or_csv, (str, Path)):
        csv_path = Path(srrs_or_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
    elif isinstance(srrs_or_csv, pd.DataFrame):
        df = srrs_or_csv
    else:
        # Iterable input (list/tuple/set/generator).
        seq = list(srrs_or_csv)
        # Deduplicate while preserving order.
        return list(dict.fromkeys(str(x).strip() for x in seq if str(x).strip()))

    # For DataFrame input: locate the column.
    cols_lower = {c.lower(): c for c in df.columns}
    col = None
    for cand in ("run", "run_accession"):
        if cand in cols_lower:
            col = cols_lower[cand]
            break
    if col is None:
        raise ValueError(
            f"CSV must contain a 'Run' (or 'run_accession') column. Got columns: {list(df.columns)}"
        )

    values = [str(x).strip() for x in df[col].tolist()]
    # Deduplicate while preserving order.
    return list(dict.fromkeys(v for v in values if v))
    
def run_pipeline(srr_list_or_csv):
    # NEW: Accept CSV / DataFrame / list.
    srr_list = _normalize_srr_list(srr_list_or_csv)
    if not srr_list:
        raise ValueError("No SRR identifiers were parsed from the input.")
    logger = _make_local_logger("PIPE")
    # Data representations handed off between steps.
    fastq_paths = []       # [(srr, fq1, fq2)]
    clean_fastqs = []      # [(srr, clean1, clean2)]
    bam_paths = []         # [(srr, bam)]
    count_tables = []      # Optional.

    # Step 0: prefetch (per SRR).
    #logger.info(f"[RUN] prefetch {srr_id} -> {output_path}")
    prefetch_step = STEPS[0]

    for srr in srr_list:
        outs = _render_paths(prefetch_step["outputs"], SRR=srr)
        # Use guarded validation.
        if _safe_call_validation(prefetch_step["validation"], outs, logger):
            logger.info(f"[SKIP] prefetch {srr}")
        else:
            ok = prefetch_step["command"](srr, logger=logger)
            if not ok:
                logger.error(f"[FAIL] prefetch {srr}")
                return False

    # Step 1: fasterq (batch mode).
    fasterq_step = STEPS[1]
    fq_products = fasterq_step["command"](srr_list, logger=logger)
    order = {s: i for i, s in enumerate(srr_list)}
    fastq_paths = []
    for rec in sorted(fq_products, key=lambda x: order.get(x[0], 0)):
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            raise ValueError(f"Unexpected fasterq record: {rec}")
        srr, fq1 = rec[0], rec[1]
        fq2 = rec[2] if len(rec) > 2 else None
        fastq_paths.append((srr, str(fq1), str(fq2) if fq2 else None))

    # Step 2: fastp (batch mode).
    fastp_step = STEPS[2]
    fp_products = fastp_step["command"](fastq_paths, logger=logger)
    order_fp = {s: i for i, (s, *_rest) in enumerate(fastq_paths)}
    clean_fastqs = []
    for rec in sorted(fp_products, key=lambda x: order_fp.get(x[0], 0)):
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            raise ValueError(f"Unexpected fastp record: {rec}")
        srr, c1 = rec[0], rec[1]
        c2 = rec[2] if len(rec) > 2 else None
        clean_fastqs.append((srr, str(c1), str(c2) if c2 else None))

    paired_flags = {srr: bool(c2) for srr, _c1, c2 in clean_fastqs}

    # Step 3: STAR (per sample; parallelizable, shown sequentially for clarity).
    star_step = STEPS[3]
    outs_by_srr = []
    for srr, c1, c2 in clean_fastqs:
        outs = _render_paths(star_step["outputs"], SRR=srr)
        outs_by_srr.append((srr, c1, c2, outs))
    
    bam_paths = []  # Structured as [(srr, bam, idx_dir)].
    for srr, c1, c2, outs in outs_by_srr:
        if star_step["validation"](outs):
            logger.info(f"[SKIP] STAR {srr}")
            # If validation succeeds, outs[0] BAM already exists and index_dir is unavailable now.
            # Safest option: set index_dir to None and infer it later (see pre-Step 4 logic).
            bam_paths.append((srr, outs[0], None))  # NEW.
        else:
            prods = star_step["command"]([(srr, c1, c2)], logger=logger)
            # Expect star_step["command"] to return [(srr, bam_path, index_dir)]  ← NEW (see notes).
            bam_paths.extend(prods)

    # Normalize to triplets (srr, bam, index_dir|None)—a key fix.
    _norm_bams = []
    for rec in bam_paths:
        if isinstance(rec, (list, tuple)):
            if len(rec) == 3:
                _norm_bams.append((rec[0], rec[1], rec[2]))
            elif len(rec) == 2:
                srr, bam = rec
                _norm_bams.append((srr, bam, None))
            else:
                raise ValueError(f"Unexpected bam_paths record (len={len(rec)}): {rec}")
        else:
            raise ValueError(f"Unexpected bam_paths record type: {type(rec)} -> {rec}")
    bam_paths = _norm_bams  # Structure now unified.
    
    # --- NEW: Before featureCounts, infer a GTF using any non-None index_dir ---
    def _find_gtf_from_index(index_dir: str | os.PathLike) -> str:
        """
        Given STAR's index_dir (typically .../index/<species>/<build>/STAR),
        search nearby cache/parent directories for extracted .gtf files.
        Rules:
          1) Look for *.gtf in the directory or its parent.
          2) Look for *.gtf under the grandparent's _cache/**.
        """
        from pathlib import Path
        idx = Path(index_dir)
        if not idx:
            return None
    
        # 1) Search *.gtf in this directory or its parent.
        for base in {idx, idx.parent}:
            for p in base.glob("*.gtf"):
                return str(p.resolve())
    
        # 2) Search the grandparent _cache directory (compatible with ensure_star_index layout).
        for base in {idx.parent, idx.parent.parent}:
            cache = base / "_cache"
            if cache.exists():
                hits = list(cache.rglob("*.gtf"))
                if hits:
                    return str(hits[0].resolve())
    
        # Fallback: look one additional level up for *.gtf.
        for p in idx.parent.parent.glob("*.gtf"):
            return str(p.resolve())
    
        return None
    
    # Pick one index_dir to infer the GTF.
    inferred_gtf = None
    for _, __, idx_dir in bam_paths:
        if idx_dir:
            inferred_gtf = _find_gtf_from_index(idx_dir)
            if inferred_gtf:
                break
    
    if not inferred_gtf:
        # When STAR entirely skips, every idx_dir might be None.
        # In that case, search the conventionally agreed locations (adjust the fallback for your layout).
        # For example, index/human/GRCh38/STAR often has a GTF in the grandparent _cache directory.
        candidate = Path("index")
        if candidate.exists():
            for p in candidate.rglob("*.gtf"):
                inferred_gtf = str(p.resolve())
                break
    
    if not inferred_gtf:
        logger.error("[featureCounts] Unable to locate a GTF automatically; ensure ensure_star_index downloaded and unpacked annotations.")
        # You can return or raise here; staying consistent with existing logic, raise an exception:
        raise RuntimeError("GTF not found. Please check STAR index/annotation preparation.")

    # Step 4: featureCounts (batch mode).
    fc_step = STEPS[4]
    outs_by_srr = []
    fc_inputs = []
    for srr, bam, _idx in bam_paths:  # Note bam_paths now contains triplets.
        outs = _render_paths(fc_step["outputs"], SRR=srr)
        outs_by_srr.append((srr, bam, outs))
        fc_inputs.append((srr, bam, paired_flags.get(srr)))

    if all(fc_step["validation"](o) for _, _, o in outs_by_srr):
        logger.info("[SKIP] featureCounts for all")
        count_tables = [o[0] for _, _, o in outs_by_srr]
    else:
        # Explicitly pass gtf to the featureCounts command function  ← NEW.
        ret = fc_step["command"](fc_inputs,
                                 logger=logger,
                                 gtf=inferred_gtf)
        count_tables = [o[0] for _, _, o in outs_by_srr]
'''
How to use
import data_prepare_pipline
srrs = ["SRR12544419", "SRR12544565", "SRR12544566"]
data_prepare_pipline.run_pipeline(srrs)
'''
