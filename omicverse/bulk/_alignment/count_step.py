# count_step.py featureCounts step
from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence

# Batch wrapper: internally calls featureCounts; simple=True keeps only gene_id,count.
from .count_tools import feature_counts_batch

def make_featurecounts_step(
    out_root: str = "work/counts",
    simple: bool = True,            # Keep only gene_id,count when True.
    gtf: str | None = None,         # Leave empty to use the GENCODE GTF downloaded by ensure_star_index.
    by: str = "auto",               # One of "srr", "accession", or "auto".
    threads: int = 8,
    gtf_path: str | None = None,
):
    """
    Input: BAM list [(srr, bam), ...]
    Output:
      - Per-sample counts: work/counts/{SRR}/{SRR}.counts.txt (or .csv)
      - Optional aggregate matrix: work/counts/matrix.{by}.csv
    Validation: per-sample count files exist and contain rows.
    """
    def _cmd(bam_pairs: Sequence[tuple[str, str]], logger=None, gtf: str | None = None):
        """
        bam_pairs: [(srr, bam_path), ...]
        gtf:       Optional runtime GTF override (takes highest priority).
        """
        os.makedirs(out_root, exist_ok=True)

        # Decide which GTF to use: runtime gtf > factory gtf_path > FC_GTF_HINT.
        gtf_use = gtf or gtf_path or os.environ.get("FC_GTF_HINT")
        if not gtf_use or not os.path.exists(gtf_use):
            raise RuntimeError(
                "[featureCounts] GTF not provided and FC_GTF_HINT not set or file missing.\n"
                f"  - gtf (runtime): {gtf}\n"
                f"  - gtf_path (factory): {gtf_path}\n"
                f"  - FC_GTF_HINT (env): {os.environ.get('FC_GTF_HINT')}"
            )

        # Invoke the batch counting helper with error handling.
        try:
            return feature_counts_batch(
                bam_items=list(bam_pairs),   # [(srr, bam)]
                out_dir=out_root,
                gtf=gtf_use,
                simple=simple,
                by=by,
                threads=threads,
                max_workers=None,            # Add parallelism here if desired.
            )
        except Exception as e:
            if logger:
                logger.error(f"[featureCounts] Batch processing failed: {e}")
            raise RuntimeError(f"featureCounts batch processing failed: {e}") from e


    return {
        "name": "featurecounts",
        "command": _cmd,  # Accepts [(srr, bam), ...].
        "outputs": [f"{out_root}" + "/{SRR}/{SRR}.counts.txt"],  # Adjust if your internal naming differs (e.g., .csv).
        "validation": lambda fs: all(os.path.exists(f) and os.path.getsize(f) > 0 for f in fs),
        "takes": "BAM_PATHS",
        "yields": "COUNT_TABLES"
    }
