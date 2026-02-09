"""STAR alignment wrapper."""
from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..utils.registry import register_function
from ._cli_utils import (
    build_env,
    ensure_dir,
    is_gz,
    parse_memory_bytes,
    resolve_executable,
    resolve_jobs,
    run_cmd,
    run_in_threads,
)


def _read_command_for_gz(auto_install: bool) -> Optional[str]:
    for name, cmd in (("pigz", "pigz -dc"), ("gzip", "gzip -dc"), ("zcat", "zcat")):
        try:
            resolve_executable(name, auto_install=auto_install)
            return cmd
        except FileNotFoundError:
            continue
    return None


def _has_genome_index(genome_dir: Path) -> bool:
    return (genome_dir / "genomeParameters.txt").exists()


def _infer_genome_fasta(genome_dir: Path) -> List[str]:
    exts = (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")
    hits: List[str] = []
    if genome_dir.exists():
        for path in genome_dir.iterdir():
            if path.is_file() and any(path.name.endswith(ext) for ext in exts):
                hits.append(str(path))
    return sorted(hits, key=lambda p: (p.endswith(".gz"), p))


def _decompress_gzip(src: Path, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)
    tmp.replace(dest)


def _ensure_uncompressed_gtf(path: str) -> str:
    src = Path(path)
    if not src.name.endswith(".gz"):
        return str(src)
    dest = src.with_suffix("")
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    _decompress_gzip(src, dest)
    return str(dest)


def _ensure_uncompressed_fasta(path: str) -> str:
    src = Path(path)
    if not src.name.endswith(".gz"):
        return str(src)
    dest = src.with_suffix("")
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    _decompress_gzip(src, dest)
    return str(dest)


def _prepare_fasta_files(
    fasta_files: Sequence[str],
    strict: bool,
) -> List[str]:
    out: List[str] = []
    for item in fasta_files:
        try:
            out.append(_ensure_uncompressed_fasta(item))
        except Exception as exc:
            if strict:
                raise
            print(f"[STAR] failed to prepare FASTA {item}: {exc}")
    return out


def _prepare_gtf_file(gtf: Optional[str], strict: bool) -> Optional[str]:
    if not gtf:
        return None
    src = Path(gtf)
    if not src.exists():
        if strict:
            raise FileNotFoundError(f"GTF not found: {gtf}")
        print(f"[STAR] GTF not found: {gtf}; continuing without GTF")
        return None
    try:
        return _ensure_uncompressed_gtf(gtf)
    except Exception as exc:
        if strict:
            raise
        print(f"[STAR] failed to prepare GTF {gtf}: {exc}")
        return None


def _build_star_index(
    genome_dir: Path,
    fasta_files: Sequence[str],
    gtf: Optional[str],
    gtf_feature: Optional[str],
    sjdb_overhang: Optional[int],
    threads: int,
    star_bin: str,
    env: dict,
    extra_args: Optional[Sequence[str]],
) -> None:
    cmd = [
        star_bin,
        "--runMode",
        "genomeGenerate",
        "--genomeDir",
        str(genome_dir),
        "--genomeFastaFiles",
    ]
    cmd.extend(list(fasta_files))
    cmd.extend(["--runThreadN", str(threads)])
    if gtf:
        cmd.extend(["--sjdbGTFfile", gtf])
        if gtf_feature:
            cmd.extend(["--sjdbGTFfeatureExon", gtf_feature])
    if sjdb_overhang is not None:
        cmd.extend(["--sjdbOverhang", str(sjdb_overhang)])
    if extra_args:
        cmd.extend(list(extra_args))
    run_cmd(cmd, env=env)


def _clean_star_outputs(sample_dir: Path) -> None:
    patterns = [
        "Aligned.*",
        "Log.*",
        "SJ.out.tab",
        "ReadsPerGene.out.tab",
        "Gene.out.tab",
        "Signal.*",
        "Unmapped.*",
        "Chimeric.*",
        "Solo.out*",
        "*STARtmp*",
    ]
    for pattern in patterns:
        for path in sample_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def _run_star_one(
    sample: str,
    fq1: Path,
    fq2: Optional[Path],
    genome_dir: Path,
    out_root: Path,
    threads: int,
    memory: str,
    gtf: Optional[str],
    sjdb_overhang: Optional[int],
    extra_args: Optional[Sequence[str]],
    star_bin: str,
    env: dict,
    auto_install: bool,
    index_ok: bool,
    strict: bool,
    overwrite: bool,
) -> Dict[str, str]:
    sample_dir = ensure_dir(out_root / sample)
    if overwrite:
        _clean_star_outputs(sample_dir)
    prefix = str(sample_dir) + os.sep
    bam_path = sample_dir / "Aligned.sortedByCoord.out.bam"

    if not index_ok:
        msg = f"STAR genome index missing in {genome_dir}. Provide genome_fasta_files or run genomeGenerate."
        if strict:
            raise FileNotFoundError(msg)
        return {"sample": sample, "error": msg}

    if not overwrite and bam_path.exists() and bam_path.stat().st_size > 1_000_000:
        return {"sample": sample, "bam": str(bam_path)}

    cmd = [
        star_bin,
        "--genomeDir",
        str(genome_dir),
        "--runThreadN",
        str(threads),
        "--outFileNamePrefix",
        prefix,
        "--outSAMtype",
        "BAM",
        "SortedByCoordinate",
        "--limitBAMsortRAM",
        str(parse_memory_bytes(memory)),
    ]

    if gtf:
        cmd.extend(["--sjdbGTFfile", gtf])
    if sjdb_overhang is not None:
        cmd.extend(["--sjdbOverhang", str(sjdb_overhang)])

    if is_gz(fq1) or (fq2 and is_gz(fq2)):
        read_cmd = _read_command_for_gz(auto_install)
        if not read_cmd:
            msg = "pigz/gzip/zcat not found for gz FASTQ input"
            if strict:
                raise FileNotFoundError(msg)
            return {"sample": sample, "error": msg}
        cmd.extend(["--readFilesCommand", read_cmd])

    cmd.extend(["--readFilesIn", str(fq1)])
    if fq2:
        cmd.append(str(fq2))

    if extra_args:
        cmd.extend(list(extra_args))

    try:
        run_cmd(cmd, env=env)
    except Exception as exc:
        if strict:
            raise
        return {"sample": sample, "error": str(exc)}

    if not bam_path.exists() or bam_path.stat().st_size == 0:
        msg = f"STAR did not produce BAM for {sample}"
        if strict:
            raise RuntimeError(msg)
        return {"sample": sample, "error": msg}

    return {"sample": sample, "bam": str(bam_path)}


@register_function(
    aliases=["STAR", "star", "align_star"],
    category="alignment",
    description="Align FASTQ reads using STAR.",
    examples=[
        "ov.alignment.STAR([('S1','S1_1.fq.gz','S1_2.fq.gz')], genome_dir='index', output_dir='star')",
    ],
    related=["alignment.fastp", "alignment.featureCount"],
)
def STAR(
    samples: Union[
        Tuple[str, str],
        Tuple[str, str, Optional[str]],
        Sequence[Tuple[str, str]],
        Sequence[Tuple[str, str, Optional[str]]],
    ],
    genome_dir: str,
    output_dir: str = "star",
    threads: int = 8,
    memory: str = "50G",
    jobs: Optional[int] = None,
    max_workers: Optional[int] = None,
    gtf: Optional[str] = None,
    gtf_feature: Optional[str] = "exon",
    sjdb_overhang: Optional[int] = None,
    genome_fasta_files: Optional[Union[str, Sequence[str]]] = None,
    auto_index: bool = True,
    strict: bool = False,
    genome_generate_threads: Optional[int] = None,
    genome_generate_sjdb_overhang: Optional[int] = None,
    genome_generate_gtf_feature: Optional[str] = None,
    genome_generate_extra_args: Optional[Sequence[str]] = None,
    extra_args: Optional[Sequence[str]] = None,
    star_path: Optional[str] = None,
    auto_install: bool = True,
    overwrite: bool = False,
) -> Union[Dict[str, str], List[Dict[str, str]]]:
    """
    Run STAR alignment.

    Parameters
    ----------
    samples
        Single sample tuple (sample, fq1[, fq2]) or list of such tuples.
    genome_dir
        STAR genome index directory.
    output_dir
        Output directory for per-sample STAR outputs.
    threads
        Threads per STAR job.
    memory
        Memory limit for BAM sorting (e.g. '50G').
    jobs
        Concurrent jobs.
    max_workers
        Legacy alias for jobs.
    gtf
        Optional GTF for splice junctions.
    gtf_feature
        GTF feature name for exons (default: exon).
    sjdb_overhang
        Optional SJDB overhang.
    genome_fasta_files
        FASTA file(s) for auto-building STAR index if missing.
    auto_index
        If True, attempt to build STAR index automatically when missing.
    strict
        If True, raise errors; otherwise return error messages per sample.
    genome_generate_threads
        Threads for genomeGenerate (defaults to threads).
    genome_generate_sjdb_overhang
        sjdbOverhang used during genomeGenerate (defaults to sjdb_overhang).
    genome_generate_gtf_feature
        GTF feature name used during genomeGenerate (defaults to gtf_feature).
    genome_generate_extra_args
        Extra args for genomeGenerate.
    extra_args
        Additional STAR CLI arguments.
    star_path
        Explicit path to STAR executable.
    auto_install
        Install missing tools automatically when possible.
    overwrite
        If True, rerun STAR and overwrite existing outputs.
    """
    if isinstance(samples, tuple):
        if len(samples) == 2:
            sample_list = [(samples[0], samples[1], None)]
        elif len(samples) == 3:
            sample_list = [samples]  # type: ignore[list-item]
        else:
            raise ValueError("samples tuple must be (sample, fq1) or (sample, fq1, fq2)")
        single_input = True
    else:
        raw_list = list(samples)  # type: ignore[arg-type]
        sample_list = []
        for item in raw_list:
            if len(item) == 2:
                sample_list.append((item[0], item[1], None))
            elif len(item) == 3:
                sample_list.append(item)  # type: ignore[list-item]
            else:
                raise ValueError("each samples item must be (sample, fq1) or (sample, fq1, fq2)")
        single_input = False

    genome_dir_path = Path(genome_dir)
    if not genome_dir_path.exists():
        if auto_index and genome_fasta_files:
            genome_dir_path = ensure_dir(genome_dir_path)
        else:
            msg = f"genome_dir not found: {genome_dir}"
            if strict:
                raise FileNotFoundError(msg)
            return {"error": msg} if single_input else [{"error": msg}]

    out_root = ensure_dir(output_dir)
    try:
        star_bin = resolve_executable("STAR", star_path, auto_install=auto_install)
    except Exception as exc:
        msg = str(exc)
        if strict:
            raise
        error_list = [{"sample": item[0], "error": msg} for item in sample_list] if sample_list else [{"error": msg}]
        return error_list[0] if single_input else error_list
    env = build_env(extra_paths=[str(Path(star_bin).parent)])

    prepared_gtf = _prepare_gtf_file(gtf, strict=strict)
    gtf_for_run = prepared_gtf or gtf

    index_ok = _has_genome_index(genome_dir_path)
    if auto_index and not index_ok:
        fasta_files: List[str]
        if genome_fasta_files is None:
            fasta_files = _infer_genome_fasta(genome_dir_path)
        elif isinstance(genome_fasta_files, str):
            fasta_files = [genome_fasta_files]
        else:
            fasta_files = list(genome_fasta_files)

        if fasta_files:
            fasta_files = _prepare_fasta_files(fasta_files, strict=strict)
            try:
                _build_star_index(
                    genome_dir=genome_dir_path,
                    fasta_files=fasta_files,
                    gtf=gtf_for_run,
                    gtf_feature=genome_generate_gtf_feature or gtf_feature,
                    sjdb_overhang=genome_generate_sjdb_overhang or sjdb_overhang,
                    threads=genome_generate_threads or threads,
                    star_bin=star_bin,
                    env=env,
                    extra_args=genome_generate_extra_args,
                )
            except Exception as exc:
                if strict:
                    raise
                print(f"[STAR] genomeGenerate failed: {exc}")
                if gtf_for_run:
                    try:
                        print("[STAR] retrying genomeGenerate without GTF")
                        _build_star_index(
                            genome_dir=genome_dir_path,
                            fasta_files=fasta_files,
                            gtf=None,
                            gtf_feature=None,
                            sjdb_overhang=genome_generate_sjdb_overhang or sjdb_overhang,
                            threads=genome_generate_threads or threads,
                            star_bin=star_bin,
                            env=env,
                            extra_args=genome_generate_extra_args,
                        )
                    except Exception as retry_exc:
                        print(f"[STAR] genomeGenerate retry without GTF failed: {retry_exc}")
        else:
            print("[STAR] genome index missing and no genome_fasta_files provided; skipping genomeGenerate")

        index_ok = _has_genome_index(genome_dir_path)

    worker_count = resolve_jobs(len(sample_list), jobs, max_workers)

    def _worker(item: Tuple[str, str, Optional[str]]) -> Dict[str, str]:
        sample, fq1, fq2 = item
        return _run_star_one(
            sample=sample,
            fq1=Path(fq1),
            fq2=Path(fq2) if fq2 else None,
            genome_dir=genome_dir_path,
            out_root=out_root,
            threads=threads,
            memory=memory,
            gtf=gtf_for_run,
            sjdb_overhang=sjdb_overhang,
            extra_args=extra_args,
            star_bin=star_bin,
            env=env,
            auto_install=auto_install,
            index_ok=index_ok,
            strict=strict,
            overwrite=overwrite,
        )

    results = run_in_threads(sample_list, _worker, worker_count)
    if single_input:
        return results[0]
    return results
