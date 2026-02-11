"""featureCounts wrapper for RNA-seq quantification."""
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Mapping

import pandas as pd

from .._registry import register_function
from ._cli_utils import (
    build_env,
    ensure_dir,
    resolve_executable,
    resolve_jobs,
    run_in_threads,
)


def _infer_paired_from_bam(bam_path: str, env: dict) -> Optional[bool]:
    try:
        import pysam  # type: ignore
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            for rec in bam.fetch(until_eof=True):
                return bool(rec.is_paired)
    except Exception:
        pass

    try:
        samtools = resolve_executable("samtools", auto_install=False)
    except Exception:
        return None

    try:
        proc = subprocess.run(
            [samtools, "view", "-c", "-f", "1", bam_path],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        paired_count = int(proc.stdout.strip() or 0)
        return paired_count > 0
    except Exception:
        return None


def _parse_gtf_attrs(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for part in raw.strip().strip(";").split(";"):
        part = part.strip()
        if not part:
            continue
        if " " in part:
            key, val = part.split(" ", 1)
            attrs[key] = val.strip().strip('"')
    return attrs


def _load_gene_map_from_gtf(gtf_path: Path, gene_name_field: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    opener = gzip.open if gtf_path.name.endswith(".gz") else open
    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in {"gene", "exon", "transcript"}:
                continue
            attrs = _parse_gtf_attrs(parts[8])
            gene_id = attrs.get("gene_id")
            if not gene_id or gene_id in mapping:
                continue
            gene_name = attrs.get(gene_name_field) or gene_id
            mapping[gene_id] = gene_name
    return mapping


def _load_gene_map(gene_map: Optional[Union[Mapping[str, str], str]], gtf_path: Path,
                   gene_name_field: str, strict: bool) -> Optional[Dict[str, str]]:
    if gene_map is None:
        try:
            return _load_gene_map_from_gtf(gtf_path, gene_name_field)
        except Exception:
            if strict:
                raise
            return None
    if isinstance(gene_map, Mapping):
        return dict(gene_map)
    path = Path(gene_map)
    if path.exists() and path.suffix in {".gtf", ".gz"}:
        try:
            return _load_gene_map_from_gtf(path, gene_name_field)
        except Exception:
            if strict:
                raise
            return None
    if path.exists():
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            if strict:
                raise
            return None
        cols = [c.lower() for c in df.columns]
        if "gene_id" in cols and gene_name_field in cols:
            gid_col = df.columns[cols.index("gene_id")]
            gname_col = df.columns[cols.index(gene_name_field)]
            return dict(zip(df[gid_col].astype(str), df[gname_col].astype(str)))
        if df.shape[1] >= 2:
            return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    if strict:
        raise FileNotFoundError(f"gene_map not found or unsupported: {gene_map}")
    return None


def _apply_gene_mapping(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    mapped = series.map(lambda gid: mapping.get(gid, gid))
    if mapped.duplicated().any():
        counts = mapped.value_counts()
        mapped = pd.Series(
            [f"{name}|{gid}" if counts.get(name, 0) > 1 else name for name, gid in zip(mapped, series)],
            index=series.index,
        )
    return mapped


def _simplify_counts(out_file: Path, sample: str, gene_map: Optional[Dict[str, str]]) -> None:
    df = pd.read_csv(out_file, sep="\t", comment="#")
    annot_cols = {"Geneid", "Chr", "Start", "End", "Strand", "Length"}
    count_cols = [c for c in df.columns if c not in annot_cols]
    if not count_cols:
        raise ValueError(f"No count columns found in {out_file}")
    counts_col = count_cols[-1]
    df_simple = df[["Geneid", counts_col]].rename(columns={"Geneid": "gene_id", counts_col: sample})
    if gene_map:
        df_simple["gene_id"] = _apply_gene_mapping(df_simple["gene_id"].astype(str), gene_map)
    df_simple.to_csv(out_file, sep="\t", index=False)


def _merge_counts_matrix(results: List[Dict[str, str]], out_root: Path, strict: bool) -> Optional[pd.DataFrame]:
    frames = []
    missing = []
    for item in results:
        counts_path = item.get("counts")
        sample = item.get("sample")
        if not counts_path or not sample:
            missing.append(sample or "unknown")
            continue
        df = pd.read_csv(counts_path, sep="\t")
        if "gene_id" not in df.columns or sample not in df.columns:
            missing.append(sample)
            continue
        frames.append(df.set_index("gene_id")[[sample]])
    if not frames:
        msg = "No valid count files to merge"
        if strict:
            raise RuntimeError(msg)
        print(f"[featureCount] {msg}")
        return None
    if missing:
        msg = f"Skipped samples with missing/invalid counts: {missing}"
        if strict:
            raise RuntimeError(msg)
        print(f"[featureCount] {msg}")
    matrix = pd.concat(frames, axis=1)
    out_path = out_root / "counts_matrix.csv"
    matrix.to_csv(out_path)
    matrix.attrs["path"] = str(out_path)
    return matrix


def _decompress_gzip(src: Path, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)
    tmp.replace(dest)


def _prepare_gtf_file(gtf: str, strict: bool) -> Optional[Path]:
    path = Path(gtf)
    if path.exists() and not path.name.endswith(".gz"):
        return path
    if not path.exists() and path.suffix == ".gtf":
        gz = path.with_suffix(path.suffix + ".gz")
        if gz.exists():
            path = gz
    if not path.exists() and path.name.endswith(".gtf.gz"):
        plain = path.with_suffix("")
        if plain.exists():
            return plain
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"GTF not found: {gtf}")
        return None
    if path.name.endswith(".gz"):
        dest = path.with_suffix("")
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        try:
            _decompress_gzip(path, dest)
            return dest
        except Exception:
            if strict:
                raise
            return None
    return path


def _build_featurecounts_cmd(
    featurecounts_bin: str,
    threads: int,
    gtf: Path,
    out_file: Path,
    bam_path: Path,
    is_paired: Optional[bool],
) -> List[str]:
    cmd = [
        featurecounts_bin,
        "-T", str(threads),
        "-a", str(gtf),
        "-o", str(out_file),
    ]
    if is_paired:
        cmd.extend(["-p", "-B", "-C"])
    cmd.append(str(bam_path))
    return cmd


def _run_featurecounts_cmd(cmd: List[str], env: dict, quiet: bool) -> tuple[bool, str]:
    if not quiet:
        print(">>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    output = proc.stdout or ""
    if output and not quiet:
        print(output, end="" if output.endswith("\n") else "\n")
    return proc.returncode == 0, output


def _run_featurecounts_one(
    sample: str,
    bam_path: Path,
    gtf: Path,
    out_root: Path,
    threads: int,
    simple: bool,
    is_paired: Optional[bool],
    featurecounts_bin: str,
    env: dict,
    strict: bool,
    auto_fix: bool,
    quiet: bool,
    gene_map: Optional[Dict[str, str]],
    overwrite: bool,
) -> Dict[str, str]:
    if not bam_path.exists():
        msg = f"BAM not found: {bam_path}"
        if strict:
            raise FileNotFoundError(msg)
        return {"sample": sample, "error": msg}

    sample_dir = ensure_dir(out_root / sample)
    out_file = sample_dir / f"{sample}.counts.txt"

    if not overwrite and out_file.exists() and out_file.stat().st_size > 0:
        return {"sample": sample, "counts": str(out_file)}

    if is_paired is None:
        is_paired = _infer_paired_from_bam(str(bam_path), env)

    first_paired = is_paired if is_paired is not None else False
    cmd = _build_featurecounts_cmd(featurecounts_bin, threads, gtf, out_file, bam_path, first_paired)
    ok, out = _run_featurecounts_cmd(cmd, env=env, quiet=quiet)
    if not ok:
        if not auto_fix:
            if strict:
                raise RuntimeError(out or "featureCounts failed")
            return {"sample": sample, "error": out or "featureCounts failed"}
        second_paired = not first_paired
        if "Paired-end reads were detected in single-end read library" in out:
            second_paired = True
        elif "Single-end reads were detected in paired-end read library" in out:
            second_paired = False
        if second_paired != first_paired:
            cmd = _build_featurecounts_cmd(featurecounts_bin, threads, gtf, out_file, bam_path, second_paired)
            ok, out = _run_featurecounts_cmd(cmd, env=env, quiet=quiet)
        if not ok:
            if strict:
                raise RuntimeError(out or "featureCounts failed")
            return {"sample": sample, "error": out or "featureCounts failed"}

    if not out_file.exists() or out_file.stat().st_size == 0:
        msg = f"featureCounts failed for {sample}"
        if strict:
            raise RuntimeError(msg)
        return {"sample": sample, "error": msg}

    if simple:
        _simplify_counts(out_file, sample, gene_map)

    return {"sample": sample, "counts": str(out_file)}


def _normalize_items(
    bam_items: Union[
        Tuple[str, str],
        Tuple[str, str, bool],
        Sequence[Tuple[str, str]],
        Sequence[Tuple[str, str, bool]],
    ]
) -> Tuple[List[Tuple[str, str, Optional[bool]]], bool]:
    if isinstance(bam_items, tuple):
        items = [bam_items]
        single_input = True
    else:
        items = list(bam_items)
        single_input = False

    normalized: List[Tuple[str, str, Optional[bool]]] = []
    for item in items:
        if len(item) == 3:
            sample, bam, is_paired = item  # type: ignore[misc]
        else:
            sample, bam = item  # type: ignore[misc]
            is_paired = None
        normalized.append((sample, bam, is_paired))

    return normalized, single_input


@register_function(
    aliases=["featureCount", "featureCounts", "gene_count"],
    category="alignment",
    description="Quantify aligned reads using featureCounts.",
    examples=[
        "ov.alignment.featureCount([('S1','S1.bam')], gtf='genes.gtf', output_dir='counts')",
    ],
    related=["alignment.STAR"],
)
def featureCount(
    bam_items: Union[Tuple[str, str], Tuple[str, str, bool], Sequence[Tuple[str, str]], Sequence[Tuple[str, str, bool]]],
    gtf: str,
    output_dir: str = "counts",
    threads: int = 8,
    jobs: Optional[int] = None,
    max_workers: Optional[int] = None,
    simple: bool = True,
    merge_matrix: bool = True,
    quiet: bool = True,
    gene_mapping: bool = False,
    gene_map: Optional[Union[Mapping[str, str], str]] = None,
    gene_name_field: str = "gene_name",
    overwrite: bool = False,
    featurecounts_path: Optional[str] = None,
    auto_install: bool = True,
    strict: bool = False,
    auto_fix: bool = True,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """
    Run featureCounts on BAM files.

    Parameters
    ----------
    bam_items
        Single tuple (sample, bam[, is_paired]) or list of tuples.
    gtf
        GTF annotation file.
    output_dir
        Output directory for count files.
    threads
        Threads per featureCounts job.
    jobs
        Concurrent jobs.
    max_workers
        Legacy alias for jobs.
    simple
        Simplify output to gene_id + counts.
    merge_matrix
        Merge per-sample count files into a single CSV matrix (multiple samples only).
    quiet
        Suppress featureCounts command output.
    gene_mapping
        Map gene_id to gene_name when exporting simplified counts/matrix.
    gene_map
        Optional mapping dict or file (TSV/CSV or GTF) for gene_id -> gene_name.
    gene_name_field
        Attribute name in GTF for gene name (default: gene_name).
    overwrite
        If True, rerun featureCounts and regenerate outputs even if files exist.
    featurecounts_path
        Explicit path to featureCounts.
    auto_install
        Install missing tools automatically when possible.
    strict
        If True, raise errors; otherwise return error messages per sample.
    auto_fix
        If True, attempt simple retries (e.g. rerun without paired flags).
    """
    item_list, single_input = _normalize_items(bam_items)

    out_root = ensure_dir(output_dir)
    gtf_path = _prepare_gtf_file(gtf, strict=strict)
    if gtf_path is None:
        msg = f"GTF not found or failed to prepare: {gtf}"
        if strict:
            raise FileNotFoundError(msg)
        error_list = [{"sample": item[0], "error": msg} for item in item_list] if item_list else [{"error": msg}]
        return error_list[0] if single_input else error_list

    try:
        featurecounts_bin = resolve_executable("featureCounts", featurecounts_path, auto_install=auto_install)
    except Exception as exc:
        msg = str(exc)
        if strict:
            raise
        error_list = [{"sample": item[0], "error": msg} for item in item_list] if item_list else [{"error": msg}]
        return error_list[0] if single_input else error_list
    env = build_env(extra_paths=[str(Path(featurecounts_bin).parent)])

    worker_count = resolve_jobs(len(item_list), jobs, max_workers)

    gene_map_obj: Optional[Dict[str, str]] = None
    if gene_mapping:
        gene_map_obj = _load_gene_map(gene_map, gtf_path, gene_name_field, strict=strict)
        if gene_map_obj is None:
            msg = "gene_mapping enabled but no mapping could be loaded"
            if strict:
                raise RuntimeError(msg)
            print(f"[featureCount] {msg}; falling back to gene_id")

    def _worker(item: Tuple[str, str, Optional[bool]]) -> Dict[str, str]:
        sample, bam, is_paired = item
        return _run_featurecounts_one(
            sample=sample,
            bam_path=Path(bam),
            gtf=gtf_path,
            out_root=out_root,
            threads=threads,
            simple=simple,
            is_paired=is_paired,
            featurecounts_bin=featurecounts_bin,
            env=env,
            strict=strict,
            auto_fix=auto_fix,
            quiet=quiet,
            gene_map=gene_map_obj if gene_mapping else None,
            overwrite=overwrite,
        )

    results = run_in_threads(item_list, _worker, worker_count)
    if not single_input and merge_matrix:
        matrix = _merge_counts_matrix(results, out_root, strict=strict)
        if matrix is not None:
            return matrix
    if single_input:
        return results[0]
    return results
