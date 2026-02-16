#!/usr/bin/env python3
"""
CLI Interface Wrapper - Run kb-python via command-line with a function-style API

This module provides a Python interface that wraps the `kb` command line tool
(kb-python), so you can call kb workflows from Python without importing
kb_python internals.

Author: Claude Code
Created: 2025
"""

import os
import sys
import shlex
import shutil
from uuid import uuid4
from typing import List, Dict, Optional, Union
import importlib.util
from .._registry import register_function
from ._cli_utils import build_env, ensure_dir, resolve_executable, run_cmd


def _resolve_kb() -> str:
    """Resolve the 'kb' executable, falling back to ``python -m kb``."""
    try:
        return resolve_executable("kb", auto_install=False)
    except FileNotFoundError:
        pass

    python_exe = sys.executable or shutil.which('python3') or shutil.which('python')
    if python_exe:
        try:
            if importlib.util.find_spec('kb') is not None:
                return f'{python_exe} -m kb'
            if importlib.util.find_spec('kb_python') is not None:
                return f'{python_exe} -m kb_python'
        except Exception:
            pass

    raise FileNotFoundError(
        "Could not find the 'kb' executable on PATH and neither 'kb' nor "
        "'kb_python' modules are importable. "
        "Please install kb-python: pip install kb-python"
    )


def _run_kb_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None,
                cwd: Optional[str] = None) -> None:
    """Run a kb command, splitting ``python -m`` prefixes when needed."""
    if isinstance(cmd[0], str) and ' ' in cmd[0]:
        first = shlex.split(cmd[0])
        cmd = first + cmd[1:]
    run_cmd(cmd, env=env, cwd=cwd)


def _resolve_parallel_fastq_dump() -> str:
    """Resolve the 'parallel-fastq-dump' executable."""
    try:
        return resolve_executable("parallel-fastq-dump", auto_install=False)
    except FileNotFoundError:
        pass

    python_exe = sys.executable or shutil.which('python3') or shutil.which('python')
    if python_exe:
        try:
            if importlib.util.find_spec('parallel-fastq-dump') is not None:
                return f'{python_exe} -m parallel-fastq-dump'
        except Exception:
            pass

    raise FileNotFoundError(
        "Could not find 'parallel-fastq-dump' on PATH. "
        "Install via: conda install -c bioconda parallel-fastq-dump"
    )


def _append_flag(cmd: List[str], flag: str, value: Optional[Union[str, int, float, bool]], as_bool: bool = False):
    """
    Utility to add flags to the CLI command.
    - as_bool: if True, interpret value as boolean; add flag if True (no value)
    """
    if as_bool:
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _normalize_list_arg(val: Optional[Union[str, List[str]]], sep: str = ',') -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, list):
        return sep.join(val)
    return val


def _include_exclude_to_flags(include: Optional[List[Dict[str, str]]],
                              exclude: Optional[List[Dict[str, str]]]) -> List[str]:
    """
    Convert include/exclude attribute dicts to CLI flags.
    Each item should be like {"attribute": "key", "pattern": "value"} or {"key": "...", "value": "..."}.
    Produces repeated flags:
      --include-attribute key:pattern
      --exclude-attribute key:pattern
    """
    out: List[str] = []
    if include:
        for item in include:
            key = item.get('attribute') or item.get('key')
            pat = item.get('pattern') or item.get('value')
            if key and pat:
                out.extend(['--include-attribute', f'{key}:{pat}'])
    if exclude:
        for item in exclude:
            key = item.get('attribute') or item.get('key')
            pat = item.get('pattern') or item.get('value')
            if key and pat:
                out.extend(['--exclude-attribute', f'{key}:{pat}'])
    return out


def ref(
    index_path: str,
    t2g_path: str,
    fasta_paths: Optional[Union[str, List[str]]] = None,
    gtf_paths: Optional[Union[str, List[str]]] = None,
    cdna_path: Optional[str] = None,
    workflow: str = 'standard',
    d: Optional[str] = None,
    k: Optional[int] = None,
    threads: int = 8,
    overwrite: bool = False,
    temp_dir: str = 'tmp',
    make_unique: bool = False,
    include: Optional[List[Dict[str, str]]] = None,
    exclude: Optional[List[Dict[str, str]]] = None,
    dlist: Optional[str] = None,
    dlist_overhang: int = 1,
    aa: bool = False,
    max_ec_size: Optional[int] = None,
    nucleus: bool = False,
    # NAC/velocity workflow specific
    f2: Optional[str] = None,
    c1: Optional[str] = None,
    c2: Optional[str] = None,
    flank: Optional[int] = None,
    # KITE workflow specific
    feature: Optional[str] = None,
    no_mismatches: bool = False,
    # Custom workflow specific
    distinguish: bool = False,
    **kwargs
) -> Dict[str, str]:
    """
    Build kallisto index and transcript-to-gene mapping via `kb ref`.
    Returns a dict with metadata and common output paths.
    """
    print(f"[kb ref] Starting ref workflow: {workflow}", flush=True)

    # If user sets nucleus=True but workflow left default, switch workflow accordingly
    if workflow == 'standard' and nucleus:
        workflow = 'nucleus'

    ensure_dir(os.path.dirname(index_path) or '.')
    ensure_dir(os.path.dirname(t2g_path) or '.')

    kb = _resolve_kb()
    cmd: List[str] = [kb, 'ref']

    # Workflow
    if workflow and workflow != 'standard':
        cmd.extend(['--workflow', workflow])

    # Choose a unique tmp dir and pass via --tmp (do NOT pre-create)
    if not temp_dir or temp_dir == 'tmp' or os.path.exists(temp_dir):
        run_tmp = f"tmp-kb-{uuid4().hex}"
    else:
        run_tmp = temp_dir
    print(f"[kb ref] Using temporary directory: {run_tmp}", flush=True)
    cmd.extend(['--tmp', run_tmp])

    # Common required outputs
    _append_flag(cmd, '-i', index_path)
    _append_flag(cmd, '-g', t2g_path)

    # Options common to most workflows
    _append_flag(cmd, '-k', k)
    _append_flag(cmd, '-t', threads)
    _append_flag(cmd, '--overwrite', overwrite, as_bool=True)
    _append_flag(cmd, '--make-unique', make_unique, as_bool=True)
    _append_flag(cmd, '--no-mismatches', no_mismatches, as_bool=True)
    _append_flag(cmd, '--aa', aa, as_bool=True)
    _append_flag(cmd, '--flank', flank)
    _append_flag(cmd, '--d-list', dlist)
    _append_flag(cmd, '--d-list-overhang', dlist_overhang)
    _append_flag(cmd, '--ec-max-size', max_ec_size)
    _append_flag(cmd, '--distinguish', distinguish, as_bool=True)

    # Include/Exclude attributes
    cmd.extend(_include_exclude_to_flags(include, exclude))

    # Pass-through optional flags for binary paths and optimizations
    if kwargs.get('kallisto'):
        _append_flag(cmd, '--kallisto', kwargs['kallisto'])
    if kwargs.get('bustools'):
        _append_flag(cmd, '--bustools', kwargs['bustools'])
    if kwargs.get('opt_off'):
        _append_flag(cmd, '--opt-off', True, as_bool=True)

    # Build positional / workflow-specific arguments
    positional: List[str] = []
    if d is not None:
        print(f"[kb ref] Using pre-built reference: {d}", flush=True)
        cmd.insert(2, '-d')
        cmd.insert(3, d)
        if cdna_path:
            _append_flag(cmd, '-f1', cdna_path)
        if f2:
            _append_flag(cmd, '-f2', f2)
        if c1:
            _append_flag(cmd, '-c1', c1)
        if c2:
            _append_flag(cmd, '-c2', c2)
    else:
        fasta_joined = _normalize_list_arg(fasta_paths, ',')
        gtf_joined = _normalize_list_arg(gtf_paths, ',')

        if workflow == 'kite':
            if cdna_path:
                _append_flag(cmd, '-f1', cdna_path)
            if feature:
                positional.append(feature)
        elif workflow in ('lamanno', 'nucleus'):
            if cdna_path:
                _append_flag(cmd, '-f1', cdna_path)
            if f2:
                _append_flag(cmd, '-f2', f2)
            if c1:
                _append_flag(cmd, '-c1', c1)
            if c2:
                _append_flag(cmd, '-c2', c2)
            if fasta_joined:
                positional.append(fasta_joined)
            if gtf_joined:
                positional.append(gtf_joined)
        elif workflow == 'nac':
            if cdna_path:
                _append_flag(cmd, '-f1', cdna_path)
            if f2:
                _append_flag(cmd, '-f2', f2)
            if c1:
                _append_flag(cmd, '-c1', c1)
            if c2:
                _append_flag(cmd, '-c2', c2)
            if fasta_joined:
                positional.append(fasta_joined)
            if gtf_joined:
                positional.append(gtf_joined)
        elif workflow == 'custom':
            if fasta_joined:
                positional.append(fasta_joined)
        else:
            if fasta_joined:
                positional.append(fasta_joined)
            if gtf_joined:
                positional.append(gtf_joined)
            if cdna_path:
                _append_flag(cmd, '-f1', cdna_path)

    cmd.extend(positional)

    env = build_env()

    try:
        _run_kb_cmd(cmd, env=env)
        print("[kb ref] ref workflow completed!", flush=True)
    except Exception as e:
        print(f"[kb ref] ref workflow failed: {e}", file=sys.stderr)
        raise
    finally:
        # Best-effort cleanup if we created a unique tmp path
        if run_tmp.startswith('tmp-kb-') and os.path.isdir(run_tmp):
            shutil.rmtree(run_tmp, ignore_errors=True)

    result: Dict[str, Union[str, int, bool, Dict[str, Union[str, int, bool]]]] = {
        'workflow': workflow,
        'technology': 'N/A',
        'parameters': {
            'threads': threads,
            'k': k,
            'overwrite': overwrite,
            'workflow_type': workflow
        },
        'index_path': index_path,
        't2g_path': t2g_path
    }
    if cdna_path:
        result['cdna_path'] = cdna_path
    return result  # type: ignore[return-value]


def count(
    index_path: str,
    t2g_path: str,
    technology: str,
    fastq_paths: Union[str, List[str]],
    output_path: str = '.',
    whitelist_path: Optional[str] = None,
    replacement_path: Optional[str] = None,
    threads: int = 8,
    memory: str = '2G',
    workflow: str = 'standard',
    overwrite: bool = False,
    temp_dir: str = 'tmp',
    # Matrix options
    tcc: bool = False,
    mm: bool = False,
    filter_barcodes: bool = False,
    filter_threshold: Optional[int] = None,
    # Output formats
    loom: bool = False,
    loom_names: Optional[Union[str, List[str]]] = None,
    h5ad: bool = False,
    cellranger: bool = False,
    gene_names: bool = False,
    report: bool = False,
    # Technology-specific parameters
    strand: Optional[str] = None,
    parity: Optional[str] = None,
    fragment_l: Optional[int] = None,
    fragment_s: Optional[int] = None,
    bootstraps: Optional[int] = None,
    # Advanced options
    em: bool = False,
    aa: bool = False,
    genomebam: bool = False,
    inleaved: bool = False,
    batch_barcodes: bool = False,
    exact_barcodes: bool = False,
    numreads: Optional[int] = None,
    store_num: bool = False,
    # Long-read options
    long_read: bool = False,
    threshold: float = 0.8,
    platform: str = 'ONT',
    # NAC/lamanno workflow specific
    c1: Optional[str] = None,
    c2: Optional[str] = None,
    nucleus: bool = False,
    # Other parameters
    **kwargs
) -> Dict[str, str]:
    """
    Generate count matrix from single-cell FASTQ files via `kb count`.
    Returns a dict with metadata and commonly generated output files if present.
    """
    print(f"[kb count] Starting count workflow: {workflow}", flush=True)
    print(f"[kb count] Technology: {technology}", flush=True)
    print(f"[kb count] Output directory: {output_path}", flush=True)

    # validate tcc/mm
    if tcc and mm:
        print("[kb count] Warning: Both tcc and mm were set; preferring --tcc.", flush=True)
        mm = False

    ensure_dir(output_path)

    # Switch to nucleus workflow if requested
    if workflow == 'standard' and nucleus:
        workflow = 'nucleus'

    kb = _resolve_kb()
    cmd: List[str] = [kb, 'count']

    # Workflow
    if workflow and workflow != 'standard':
        cmd.extend(['--workflow', workflow])

    # Choose a unique tmp dir and pass via --tmp (do NOT pre-create)
    if not temp_dir or temp_dir == 'tmp' or os.path.exists(temp_dir):
        run_tmp = f"tmp-kb-{uuid4().hex}"
    else:
        run_tmp = temp_dir
    print(f"[kb count] Using temporary directory: {run_tmp}", flush=True)
    cmd.extend(['--tmp', run_tmp])

    # Required basic args
    _append_flag(cmd, '-i', index_path)
    _append_flag(cmd, '-g', t2g_path)
    _append_flag(cmd, '-x', technology)
    _append_flag(cmd, '-o', output_path)

    # c1/c2 (velocity-type workflows)
    _append_flag(cmd, '-c1', c1)
    _append_flag(cmd, '-c2', c2)

    # Optional core args
    _append_flag(cmd, '-t', threads)
    _append_flag(cmd, '-m', memory)
    _append_flag(cmd, '--overwrite', overwrite, as_bool=True)

    # Filtering (ONLY for barcode-based single-cell tech)
    is_bulk = str(technology).upper() == "BULK"
    if is_bulk:
        # BULK has no barcodes/UMIs; kb CLI rejects --filter/--filter-threshold
        if filter_barcodes:
            raise ValueError("filter_barcodes/--filter is not supported for technology='BULK'")
        if filter_threshold is not None:
            raise ValueError("filter_threshold/--filter-threshold is not supported for technology='BULK'")
    else:
        if filter_barcodes:
            cmd.extend(['--filter', 'bustools'])
        if filter_threshold is not None:
            _append_flag(cmd, '--filter-threshold', filter_threshold)

    # Matrix mode
    _append_flag(cmd, '--tcc', tcc, as_bool=True)
    if not tcc:
        _append_flag(cmd, '--mm', mm, as_bool=True)

    # Output format flags
    _append_flag(cmd, '--loom', loom, as_bool=True)
    if loom_names:
        ln = loom_names if isinstance(loom_names, str) else ','.join(loom_names)
        _append_flag(cmd, '--loom-names', ln)
    _append_flag(cmd, '--h5ad', h5ad, as_bool=True)
    _append_flag(cmd, '--cellranger', cellranger, as_bool=True)
    _append_flag(cmd, '--gene-names', gene_names, as_bool=True)
    _append_flag(cmd, '--report', report, as_bool=True)

    # gzip defaults on with cellranger unless --no-gzip specified
    if kwargs.get('no_gzip'):
        _append_flag(cmd, '--no-gzip', True, as_bool=True)
    else:
        if cellranger or kwargs.get('gzip'):
            _append_flag(cmd, '--gzip', True, as_bool=True)

    # Velocity / nac extras
    _append_flag(cmd, '-c1', c1)
    _append_flag(cmd, '-c2', c2)

    # Tech-specific
    if parity:
        _append_flag(cmd, '--parity', parity)
    if strand:
        _append_flag(cmd, '--strand', strand)
    _append_flag(cmd, '--fragment-l', fragment_l)
    _append_flag(cmd, '--fragment-s', fragment_s)
    _append_flag(cmd, '--bootstraps', bootstraps)

    # Advanced toggles
    _append_flag(cmd, '--em', em, as_bool=True)
    _append_flag(cmd, '--aa', aa, as_bool=True)
    if genomebam:
        print("[kb count] Warning: --genomebam is not supported in many kb versions.", flush=True)
    _append_flag(cmd, '--genomebam', genomebam, as_bool=True)
    _append_flag(cmd, '--inleaved', inleaved, as_bool=True)
    _append_flag(cmd, '--batch-barcodes', batch_barcodes, as_bool=True)
    _append_flag(cmd, '--exact-barcodes', exact_barcodes, as_bool=True)

    # Read count / BUS number switches
    if numreads is not None:
        _append_flag(cmd, '-N', numreads)
    _append_flag(cmd, '--num', store_num, as_bool=True)

    # Long-read options
    if long_read:
        _append_flag(cmd, '--long', True, as_bool=True)
        _append_flag(cmd, '--threshold', threshold)
        _append_flag(cmd, '--platform', platform)
    if kwargs.get('error_rate') is not None:
        _append_flag(cmd, '--error-rate', kwargs['error_rate'])

    # Additional inputs
    if whitelist_path:
        _append_flag(cmd, '-w', whitelist_path)
    if replacement_path:
        _append_flag(cmd, '-r', replacement_path)

    # Pass-through params commonly used
    if kwargs.get('kallisto'):
        _append_flag(cmd, '--kallisto', kwargs['kallisto'])
    if kwargs.get('bustools'):
        _append_flag(cmd, '--bustools', kwargs['bustools'])
    if kwargs.get('opt_off'):
        _append_flag(cmd, '--opt-off', True, as_bool=True)
    if kwargs.get('dry_run'):
        _append_flag(cmd, '--dry-run', True, as_bool=True)
    if kwargs.get('no_inspect'):
        _append_flag(cmd, '--no-inspect', True, as_bool=True)
    if kwargs.get('delete_bus'):
        _append_flag(cmd, '--delete-bus', True, as_bool=True)

    # Rare/hidden toggles
    passthrough_bool = [
        'matrix_to_files', 'matrix_to_directories', 'no_fragment',
        'union', 'no_jump', 'quant_umis', 'keep_flags'
    ]
    for key in passthrough_bool:
        if key in kwargs and kwargs[key]:
            _append_flag(cmd, f"--{key.replace('_','-')}", True, as_bool=True)

    # Scalar extras
    if kwargs.get('gtf'):
        _append_flag(cmd, '--gtf', kwargs['gtf'])
    if kwargs.get('chromosomes'):
        _append_flag(cmd, '--chromosomes', kwargs['chromosomes'])
    if kwargs.get('sum'):
        _append_flag(cmd, '--sum', kwargs['sum'])

    # Append FASTQ paths last
    fastqs = [fastq_paths] if isinstance(fastq_paths, str) else list(fastq_paths)
    cmd.extend(fastqs)

    env = build_env()

    try:
        _run_kb_cmd(cmd, env=env)
        print("[kb count] count workflow completed!", flush=True)
    except Exception as e:
        print(f"[kb count] count workflow failed: {e}", file=sys.stderr)
        raise
    finally:
        if run_tmp.startswith('tmp-kb-') and os.path.isdir(run_tmp):
            shutil.rmtree(run_tmp, ignore_errors=True)

    output_info: Dict[str, Union[str, bool, int, Dict[str, Union[str, bool, int]]]] = {
        'workflow': workflow,
        'technology': technology,
        'output_path': output_path,
        'parameters': {
            'threads': threads,
            'memory': memory,
            'filter_barcodes': filter_barcodes,
            'h5ad': h5ad,
            'loom': loom,
            'cellranger': cellranger,
            'tcc': tcc,
            'mm': mm
        }
    }

    def _maybe(path: str, key: str):
        fpath = os.path.join(output_path, path)
        if os.path.exists(fpath):
            output_info[key] = fpath  # type: ignore[index]

    _maybe("adata.h5ad", "h5ad_file")
    _maybe("adata.loom", "loom_file")
    if cellranger:
        cr_dir = os.path.join(output_path, "cellranger")
        if os.path.isdir(cr_dir):
            output_info['cellranger_dir'] = cr_dir  # type: ignore[index]
    _maybe("matrix.mtx", "matrix_file")
    _maybe("barcodes.tsv", "barcodes_file")
    _maybe("genes.tsv", "genes_file")

    return output_info  # type: ignore[return-value]


def analyze_10x_v3_data(
    fastq_files: Union[str, List[str]],
    reference_output_dir: str = "reference",
    analysis_output_dir: str = "analysis",
    threads_ref: int = 8,
    threads_count: int = 2,
    download_reference: bool = True,
    **kwargs
) -> Dict[str, str]:
    """
    One-click 10x v3 data analysis using the kb CLI: Download reference + Count analysis.
    """
    results: Dict[str, Dict[str, Union[str, int, bool]]] = {}

    ensure_dir(reference_output_dir)
    ensure_dir(analysis_output_dir)

    if download_reference:
        print("[kb] Step 1: Downloading human reference genome...", flush=True)
        ref_result = ref(
            index_path=os.path.join(reference_output_dir, "index.idx"),
            t2g_path=os.path.join(reference_output_dir, "t2g.txt"),
            d='human',
            cdna_path=os.path.join(reference_output_dir, "transcriptome.fasta"),
            threads=threads_ref
        )
        results['reference'] = ref_result  # type: ignore[index]
        print("[kb] Reference genome download complete!", flush=True)
        index_file = os.path.join(reference_output_dir, "index.idx")
        t2g_file = os.path.join(reference_output_dir, "t2g.txt")
    else:
        index_file = os.path.join(reference_output_dir, "index.idx")
        t2g_file = os.path.join(reference_output_dir, "t2g.txt")

    print("[kb] Step 2: Performing count analysis...", flush=True)
    count_result = count(
        fastq_paths=fastq_files,
        index_path=index_file,
        t2g_path=t2g_file,
        output_path=analysis_output_dir,
        technology='10XV3',
        threads=threads_count,
        **kwargs
    )
    results['count'] = count_result  # type: ignore[index]
    print("[kb] Count analysis complete!", flush=True)

    return results  # type: ignore[return-value]


@register_function(
    aliases=["parallel_fastq_dump", "并行下载SRA", "pfastq_dump"],
    category="alignment",
    description="Download SRA data in parallel using parallel-fastq-dump",
    examples=[
        "# Download SRA data with 4 threads",
        "ov.alignment.parallel_fastq_dump(sra_id='SRR2244401', threads=4, outdir='out/', split_files=True, gzip=True)",
        "# Download with specific spot range",
        "ov.alignment.parallel_fastq_dump(sra_id='SRR2244401', threads=4, outdir='out/', min_spot_id=1, max_spot_id=10000, split_files=True)"
    ],
    related=["alignment.ref", "alignment.count"]
)
def parallel_fastq_dump(
    sra_id: str,
    threads: int = 1,
    outdir: str = '.',
    tmpdir: Optional[str] = None,
    min_spot_id: int = 1,
    max_spot_id: Optional[int] = None,
    split_files: bool = False,
    gzip: bool = False,
    **kwargs
) -> Dict[str, Union[str, int]]:
    r"""Download SRA data in parallel using parallel-fastq-dump.

    This function wraps the parallel-fastq-dump tool to download sequencing data
    from NCBI SRA (Sequence Read Archive) in parallel for faster downloads.

    Arguments:
        sra_id: SRA accession ID (e.g., 'SRR2244401').
        threads: Number of threads to use for parallel download. Default: 1.
        outdir: Output directory for downloaded FASTQ files. Default: '.'.
        tmpdir: Temporary directory for intermediate files. Default: None.
        min_spot_id: Minimum spot ID to download. Default: 1.
        max_spot_id: Maximum spot ID to download. Default: None (all spots).
        split_files: Split paired-end reads into separate files. Default: False.
        gzip: Compress output files with gzip. Default: False.
        **kwargs: Additional arguments to pass to parallel-fastq-dump.

    Returns:
        result: Dictionary containing download metadata including sra_id, threads,
                outdir, and output file paths.

    Examples:
        >>> import omicverse as ov
        >>> # Download SRA data with 4 threads and split files
        >>> result = ov.alignment.parallel_fastq_dump(
        ...     sra_id='SRR2244401',
        ...     threads=4,
        ...     outdir='fastq_output/',
        ...     split_files=True,
        ...     gzip=True
        ... )
        >>> # Download with spot range limit
        >>> result = ov.alignment.parallel_fastq_dump(
        ...     sra_id='SRR2244401',
        ...     threads=8,
        ...     outdir='fastq_output/',
        ...     min_spot_id=1,
        ...     max_spot_id=100000,
        ...     split_files=True,
        ...     gzip=True
        ... )
    """
    print(f"[parallel-fastq-dump] Starting for {sra_id}", flush=True)
    print(f"[parallel-fastq-dump] Threads: {threads}", flush=True)
    print(f"[parallel-fastq-dump] Output directory: {outdir}", flush=True)

    # Ensure output directory exists
    ensure_dir(outdir)

    # Check if parallel-fastq-dump is available
    pfastq_dump = _resolve_parallel_fastq_dump()

    # Build command
    cmd: List[str] = [pfastq_dump]

    # Required arguments
    _append_flag(cmd, '--sra-id', sra_id)
    _append_flag(cmd, '--threads', threads)
    _append_flag(cmd, '--outdir', outdir)

    # Optional arguments
    if tmpdir:
        _append_flag(cmd, '--tmpdir', tmpdir)
    _append_flag(cmd, '--minSpotId', min_spot_id)
    if max_spot_id:
        _append_flag(cmd, '--maxSpotId', max_spot_id)

    # Boolean flags
    if split_files:
        cmd.append('--split-files')
    if gzip:
        cmd.append('--gzip')

    # Pass-through additional arguments
    for key, value in kwargs.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            _append_flag(cmd, flag, value)

    # Ensure the directory containing parallel-fastq-dump is in PATH
    env = build_env()
    pfastq_dump_dir = os.path.dirname(pfastq_dump) if ' ' not in pfastq_dump else os.path.dirname(shlex.split(pfastq_dump)[-1])
    if pfastq_dump_dir and os.path.isdir(pfastq_dump_dir):
        env['PATH'] = pfastq_dump_dir + os.pathsep + env.get('PATH', '')

    try:
        run_cmd(cmd, env=env)
        print("[parallel-fastq-dump] Completed successfully!", flush=True)
    except Exception as e:
        print(f"[parallel-fastq-dump] Failed: {e}", file=sys.stderr)
        raise

    # Build result dictionary
    result: Dict[str, Union[str, int, List[str]]] = {
        'sra_id': sra_id,
        'threads': threads,
        'outdir': outdir,
        'split_files': split_files,
        'gzip': gzip
    }

    # Detect output files
    extension = '.fastq.gz' if gzip else '.fastq'
    if split_files:
        # For paired-end data with split files
        file1 = os.path.join(outdir, f"{sra_id}_1{extension}")
        file2 = os.path.join(outdir, f"{sra_id}_2{extension}")
        if os.path.exists(file1):
            result['output_files'] = [file1]
            if os.path.exists(file2):
                result['output_files'].append(file2)  # type: ignore[union-attr]
    else:
        # Single file output
        output_file = os.path.join(outdir, f"{sra_id}{extension}")
        if os.path.exists(output_file):
            result['output_file'] = output_file

    return result  # type: ignore[return-value]


# Optional namespace for convenient access
import types
single = types.SimpleNamespace()
single.ref = ref
single.count = count
single.analyze_10x_v3_data = analyze_10x_v3_data
single.parallel_fastq_dump = parallel_fastq_dump