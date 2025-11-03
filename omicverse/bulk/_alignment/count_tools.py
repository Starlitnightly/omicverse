# count_tools.py featureCounts batch utilities
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from datetime import datetime


def _feature_counts_one_with_path(bam_path: str, out_dir: str, gtf: str, threads: int = 8, simple: bool = True, featurecounts_path: str = None):
    """Helper function that accepts a pre-resolved featureCounts path"""
    if featurecounts_path is None:
        return _feature_counts_one(bam_path, out_dir, gtf, threads, simple)

    # Use the provided path directly
    srr = Path(bam_path).stem.replace(".bam", "")
    out_prefix = Path(out_dir) / srr
    out_file = f"{out_prefix}.counts.txt"

    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return srr, out_file

    cmd = [
        featurecounts_path,
        "-T", str(threads),
        "-a", gtf,
        "-o", out_file,
        bam_path
    ]

    # Use proper environment
    from .tools_check import merged_env
    env = merged_env()

    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"[featureCounts] Failed to execute featureCounts (path: {featurecounts_path})\n"
            f"Exists: {os.path.exists(featurecounts_path)}\n"
            f"Executable: {os.access(featurecounts_path, os.X_OK)}\n"
            f"Original error: {e}"
        ) from e

    # Simplify output by keeping only the count column.
    if simple and os.path.exists(out_file):
        df = pd.read_csv(out_file, sep="\t", comment="#")
        # Annotation columns produced by featureCounts.
        annot_cols = {"Geneid", "Chr", "Start", "End", "Strand", "Length"}
        # Identify the count column (usually one column named after the BAM).
        count_cols = [c for c in df.columns if c not in annot_cols]
        if len(count_cols) == 0:
            raise ValueError(f"No count columns found in {out_file}. Got columns: {list(df.columns)}")
        if len(count_cols) > 1:
            # For multiple BAMs, select the last column as a safeguard.
            counts_col = count_cols[-1]
        else:
            counts_col = count_cols[0]

        df_simple = df[["Geneid", counts_col]].rename(
            columns={"Geneid": "gene_id", counts_col: srr}
        )
        df_simple.to_csv(out_file, sep="\t", index=False)

    return srr, out_file

def _feature_counts_one(bam_path: str, out_dir: str, gtf: str, threads: int = 8, simple: bool = True):
    # -------------- Safety guard for missing GTF --------------
    if gtf is None:
        gtf = os.environ.get("FC_GTF_HINT")
    if not gtf or not os.path.exists(gtf):
        raise RuntimeError(
            f"[featureCounts] Missing valid GTF file for {bam_path}. "
            "Ensure the pipeline sets FC_GTF_HINT or pass gtf explicitly."
        )
    # -----------------------------------------

    srr = Path(bam_path).stem.replace(".bam", "")
    out_prefix = Path(out_dir) / srr
    out_file = f"{out_prefix}.counts.txt"

    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return srr, out_file

    # -------------- Enhanced featureCounts detection --------------
    from .tools_check import resolve_tool, merged_env
    import shutil

    featurecounts_path = resolve_tool("featureCounts")
    if not featurecounts_path:
        # Detailed diagnostic information for the environment.
        env_path = os.environ.get("PATH", "Not set")
        jupyter_kernel_path = os.path.dirname(sys.executable) if hasattr(sys, 'executable') else "Unknown"

        raise RuntimeError(
            f"[featureCounts] Unable to locate the featureCounts executable.\n"
            f"This often indicates the Jupyter Lab kernel environment differs from the shell.\n\n"
            f"Diagnostics:\n"
            f"  - Current Python: {sys.executable}\n"
            f"  - Jupyter kernel path: {jupyter_kernel_path}\n"
            f"  - PATH contains 'omicverse': {'omicverse' in env_path}\n"
            f"  - shutil.which('featureCounts'): {shutil.which('featureCounts')}\n\n"
            f"Suggested fixes:\n"
            f"  1. In a Jupyter cell run: !which featureCounts\n"
            f"  2. If missing, install: !conda install -c bioconda subread -y\n"
            f"  3. Or prepend the env bin: os.environ['PATH'] = '/path/to/conda/envs/omicverse/bin:' + os.environ['PATH']\n\n"
            f"Note: 'No such file or directory' refers to the missing featureCounts binary,\n"
            f"      not the directory; do not create directories manually."
        )

    # Use the resolved absolute path rather than relying on PATH.
    cmd = [
        featurecounts_path,
        "-T", str(threads),
        "-a", gtf,
        "-o", out_file,
        bam_path
    ]

    # Use the merged environment to ensure featureCounts is discoverable.
    env = merged_env()

    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as e:
        # Provide more detail if execution still fails.
        raise RuntimeError(
            f"[featureCounts] Execution failed even though the path was resolved.\n"
            f"Attempted path: {featurecounts_path}\n"
            f"Exists: {os.path.exists(featurecounts_path)}\n"
            f"Executable: {os.access(featurecounts_path, os.X_OK)}\n"
            f"Original error: {e}"
        ) from e
    
    # Simplify the output by retaining the gene/count columns.
    if simple and os.path.exists(out_file):
        df = pd.read_csv(out_file, sep="\t", comment="#")
        # Annotation columns included in featureCounts output.
        annot_cols = {"Geneid", "Chr", "Start", "End", "Strand", "Length"}
        # Identify the count column (typically the BAM name/path).
        count_cols = [c for c in df.columns if c not in annot_cols]
        if len(count_cols) == 0:
            raise ValueError(f"No count columns found in {out_file}. Got columns: {list(df.columns)}")
        if len(count_cols) > 1:
            # For multi-BAM tables, fall back to the last column.
            counts_col = count_cols[-1]
        else:
            counts_col = count_cols[0]
    
        df_simple = df[["Geneid", counts_col]].rename(
            columns={"Geneid": "gene_id", counts_col: srr}
        )
        df_simple.to_csv(out_file, sep="\t", index=False)
    
    return srr, out_file


def feature_counts_batch(
    bam_items: list[tuple[str, str]],  # [(srr, bam_path)]
    out_dir: str,
    gtf: str | None = None,
    simple: bool = True,
    by: str = "auto",
    threads: int = 8,
    max_workers: int | None = None
) -> dict[str, object]:
    """
    Run featureCounts on multiple BAM files.
    """
    os.makedirs(out_dir, exist_ok=True)
    # -------------- Safety guard for missing GTF --------------
    if gtf is None:
        gtf = os.environ.get("FC_GTF_HINT")
    if not gtf or not os.path.exists(gtf):
        raise RuntimeError(
            "[featureCounts_batch] GTF not provided and FC_GTF_HINT not found. "
            "Ensure the pipeline infers the GTF or pass one explicitly."
        )

    # -------------- Enhanced featureCounts detection --------------
    from .tools_check import resolve_tool, merged_env
    import shutil

    featurecounts_path = resolve_tool("featureCounts")
    if not featurecounts_path:
        # Detailed diagnostic information.
        env_path = os.environ.get("PATH", "Not set")
        jupyter_kernel_path = os.path.dirname(sys.executable) if hasattr(sys, 'executable') else "Unknown"

        raise RuntimeError(
            f"[featureCounts_batch] Unable to locate the featureCounts executable.\n"
            f"This often indicates the Jupyter Lab kernel environment differs from the shell.\n\n"
            f"Diagnostics:\n"
            f"  - Current Python: {sys.executable}\n"
            f"  - Jupyter kernel path: {jupyter_kernel_path}\n"
            f"  - PATH contains 'omicverse': {'omicverse' in env_path}\n"
            f"  - shutil.which('featureCounts'): {shutil.which('featureCounts')}\n\n"
            f"Suggested fixes:\n"
            f"  1. In a Jupyter cell run: !which featureCounts\n"
            f"  2. If missing, install: !conda install -c bioconda subread -y\n"
            f"  3. Or prepend the env bin: os.environ['PATH'] = '/path/to/conda/envs/omicverse/bin:' + os.environ['PATH']\n\n"
            f"Note: 'No such file or directory' refers to the missing featureCounts binary,\n"
            f"      not the directory; avoid creating directories manually."
        )
    # -----------------------------------------

    results, errors = [], []

    # Compute a reasonable worker count to avoid resource contention.
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        # Ensure each worker has enough CPU resources.
        max_workers = min(4, cpu_count // max(1, threads // 4))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_feature_counts_one_with_path, bam, out_dir, gtf, threads, simple, featurecounts_path): srr
            for srr, bam in bam_items
        }
        for fut in as_completed(futures):
            srr = futures[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"[ERR] {srr}: {e}")

    # Aggregate results; support Geneid or gene_id and rename count columns to SRR IDs.
    pairs = [(srr, f) for (srr, f) in results if os.path.exists(f)]
    if len(pairs) > 1:
        merged_df = None
        for srr, f in pairs:
            # Skip comment lines to avoid header interference.
            df = pd.read_csv(f, sep="\t", comment="#")

            # Choose gene column: prefer gene_id, otherwise Geneid, otherwise the first column.
            if "gene_id" in df.columns:
                gene_col = "gene_id"
            elif "Geneid" in df.columns:
                gene_col = "Geneid"
            else:
                gene_col = df.columns[0]

            # Determine count columns by removing annotation fields (typically leaves one column).
            meta_cols = {"Chr", "Start", "End", "Strand", "Length", gene_col}
            count_cols = [c for c in df.columns if c not in meta_cols]
            if not count_cols:
                # Fallback: treat the final column as the count column.
                count_col = df.columns[-1]
            else:
                # If multiple remain, take the last one (usually the counts).
                count_col = count_cols[-1]

            # Retain only gene_id plus that sample's count column, renaming the column to the SRR.
            df_simple = df[[gene_col, count_col]].copy()
            df_simple.columns = ["gene_id", srr]

            if merged_df is None:
                merged_df = df_simple
            else:
                merged_df = merged_df.merge(df_simple, on="gene_id", how="outer")

        # Each merged count column now matches its SRR, so no further renaming is required.
        out_path = Path(out_dir) / f"matrix.{by}.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"[OK] featureCounts merged matrix → {out_path}")

    return {
        "tables": results,                 # e.g. [(srr, sample_table_path), ...] preserving the current structure
        "matrix": str(out_path) if 'out_path' in locals() else None,
        "failed": errors if 'errors' in locals() else [],
    }

def run_featurecounts_auto(
    bam_files: list[str | Path],
    index_dir: str | Path,
    out_dir: str = "results",
    accession_id: str | None = None,   # e.g. "GSE157103"
    srr_id: str | None = None,         # e.g. "SRR12544419" for single-sample runs
    threads: int = 12,
    output_csv: bool = True,
    simple: bool = True,                # Whether to trim the output table
    featurecounts_bin: str | None = None,  # Optional explicit featureCounts path
) -> Path:
    """
    - Auto-detect the GTF from the STAR index (decompress .gtf.gz if required).
    - Auto-name outputs using the accession or SRR.
    - Convert the default featureCounts TSV output into CSV.
    """
    bam_files = [str(Path(b)) for b in bam_files]
    index_dir = Path(index_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Auto-discover the GTF.
    gtf_path = get_gtf_for_index(index_dir)  # No manual GTF path required.

    # 2) Smart naming.
    if accession_id and not srr_id:
        prefix = f"{accession_id}_counts"
    elif srr_id:
        prefix = f"{srr_id}_counts"
    else:
        prefix = f"counts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_txt = Path(out_dir) / f"{prefix}.txt"   # Let featureCounts emit its default TSV first.

    # 3) Locate the featureCounts executable.
    if featurecounts_bin is None:
        import shutil
        featurecounts_bin = shutil.which("featureCounts") or "featureCounts"

    # 4) Run featureCounts.
    cmd = [
        featurecounts_bin,
        "-T", str(threads),
        "-t", "exon",
        "-g", "gene_id",
        "-a", str(gtf_path),
        "-o", str(out_txt),
    ] + bam_files
    print(">>", " ".join(cmd))
    # Use proper environment to ensure featureCounts is found
    from .tools_check import merged_env
    env = merged_env()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)

    # Read the output.
    df = pd.read_csv(out_txt, sep="\t", comment="#")

    # When simple=True, keep only the Geneid and count columns.
    if simple:
        gene_col = "Geneid" if "Geneid" in df.columns else df.columns[0]
        # Remove annotation columns, retaining gene and sample counts for downstream alignment formatting.
        keep_cols = [gene_col] + [
            c for c in df.columns
            if c not in ["Chr", "Start", "End", "Strand", "Length"] and c != gene_col
        ]
        df = df[keep_cols]
        print(f"[INFO] Simplified output: retained {len(keep_cols)} columns")

    # Export to CSV or TXT.
    if output_csv:
        out_csv = out_txt.with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] featureCounts output → {out_csv}")
        return out_csv.resolve()
    else:
        df.to_csv(out_txt, sep="\t", index=False)
        print(f"[OK] featureCounts output → {out_txt}")
        return out_txt.resolve()
