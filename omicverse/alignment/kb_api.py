#!/usr/bin/env python3
"""
Internal API Interface - Use kb_python functions directly instead of subprocess

This module provides a direct Python interface by importing kb_python internal functions,
avoiding subprocess calls and circular import issues.

Author: Claude Code
Created: 2025
"""

import os
import sys
from typing import List, Dict, Optional, Union

class Colors:
    """ANSI color codes for terminal output styling."""
    HEADER = '\033[95m'    # Purple
    BLUE = '\033[94m'      # Blue
    CYAN = '\033[96m'      # Cyan
    GREEN = '\033[92m'     # Green
    WARNING = '\033[93m'   # Yellow
    FAIL = '\033[91m'      # Red
    ENDC = '\033[0m'       # Reset
    BOLD = '\033[1m'       # Bold
    UNDERLINE = '\033[4m'  # Underline
    
# Add kb_python package to path
#kb_python_path = os.path.join(os.path.dirname(__file__), 'kb_python')
#if kb_python_path not in sys.path:
#    sys.path.insert(0, kb_python_path)

# Lazily import kb_python modules to avoid circular imports
def _import_kb_python_modules():
    """Lazily import kb_python modules"""
    try:
        # Import ref module functions
        from ..external.kb_python.ref import (
            ref as _ref,
            ref_nac as _ref_nac,
            ref_lamanno as _ref_lamanno,
            ref_kite as _ref_kite,
            ref_custom as _ref_custom,
            download_reference as _download_reference
        )

        # Import count module functions
        from ..external.kb_python.count import (
            count as _count,
            count_nac as _count_nac,
            count_velocity as _count_velocity,
            count_velocity_smartseq3 as _count_velocity_smartseq3
        )

        # Import utility functions
        from ..external.kb_python.utils import make_directory, remove_directory

        return {
            'ref': (_ref, _ref_nac, _ref_lamanno, _ref_kite, _ref_custom, _download_reference),
            'count': (_count, _count_nac, _count_velocity, _count_velocity_smartseq3),
            'utils': (make_directory, remove_directory)
        }
    except ImportError as e:
        # 使用 print 输出错误到 stderr
        error_msg = (
            f"{Colors.FAIL}✗ Failed to import kb_python modules: {e}\n"
            "Please ensure:\n"
            "1. The kb_python package is installed correctly\n"
            "2. The current directory contains the kb_python folder\n"
            f"3. All dependencies are installed{Colors.ENDC}"
        )
        print(error_msg, file=sys.stderr) # <--- 改为 print
        raise ImportError("Failed to import kb_python modules.")


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
    # NAC workflow specific
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
    Build kallisto index and transcript-to-gene mapping - using internal API
    ... (docstring unchanged) ...
    """
    print(f"{Colors.BOLD}{Colors.HEADER}🚀 Starting ref workflow: {workflow}{Colors.ENDC}")

    # Import kb_python modules
    modules = _import_kb_python_modules()
    _ref, _ref_nac, _ref_lamanno, _ref_kite, _ref_custom, _download_reference = modules['ref']
    make_directory, remove_directory = modules['utils']

    # Ensure temporary directory exists
    make_directory(temp_dir)

    # Convert string paths to lists
    if isinstance(fasta_paths, str):
        fasta_paths = fasta_paths.split(',')
    if isinstance(gtf_paths, str):
        gtf_paths = gtf_paths.split(',')

    try:
        if d is not None:
            print(f"{Colors.BLUE}    Downloading pre-built reference genome: {d}{Colors.ENDC}")
            files = {'i': index_path, 'g': t2g_path}
            if cdna_path:
                files['f1'] = cdna_path
            if f2:
                files['f2'] = f2
            if c1:
                files['c1'] = c1
            if c2:
                files['c2'] = c2

            result = _download_reference(
                species=d,
                workflow=workflow,
                files=files,
                overwrite=overwrite,
                temp_dir=temp_dir,
                k=31 if k is None else k
            )

        elif workflow == 'nac':
            print(f"{Colors.BLUE}    Executing NAC workflow{Colors.ENDC}")
            result = _ref_nac(
                fasta_paths, gtf_paths, f2, c1, c2,
                index_path, t2g_path,
                k=k, flank=flank, include=include, exclude=exclude,
                threads=threads, dlist=dlist, dlist_overhang=dlist_overhang,
                overwrite=overwrite, make_unique=make_unique,
                temp_dir=temp_dir, max_ec_size=max_ec_size
            )

        elif workflow == 'lamanno':
            print(f"{Colors.BLUE}    Executing lamanno workflow{Colors.ENDC}")
            result = _ref_lamanno(
                fasta_paths, gtf_paths, cdna_path, f2,
                index_path, t2g_path, c1, c2,
                k=k, flank=flank, include=include, exclude=exclude,
                overwrite=overwrite, temp_dir=temp_dir, threads=threads
            )

        elif workflow == 'nucleus':
            print(f"{Colors.BLUE}    Executing nucleus workflow{Colors.ENDC}")
            result = _ref(
                fasta_paths, gtf_paths, cdna_path, index_path, t2g_path,
                nucleus=True, k=k, include=include, exclude=exclude,
                overwrite=overwrite, temp_dir=temp_dir, threads=threads,
                make_unique=make_unique, dlist=dlist, dlist_overhang=dlist_overhang,
                aa=aa, max_ec_size=max_ec_size
            )

        elif workflow == 'kite':
            print(f"{Colors.BLUE}    Executing KITE workflow{Colors.ENDC}")
            result = _ref_kite(
                feature, cdna_path, index_path, t2g_path,
                k=k, no_mismatches=no_mismatches, threads=threads,
                overwrite=overwrite, temp_dir=temp_dir
            )

        elif workflow == 'custom':
            print(f"{Colors.BLUE}    Executing Custom workflow{Colors.ENDC}") 
            result = _ref_custom(
                fasta_paths, index_path, k=k, threads=threads,
                dlist=dlist, dlist_overhang=dlist_overhang, aa=aa,
                overwrite=overwrite, temp_dir=temp_dir,
                make_unique=make_unique, distinguish=distinguish
            )

        else:
            print(f"{Colors.BLUE}    Executing standard workflow{Colors.ENDC}") 
            result = _ref(
                fasta_paths, gtf_paths, cdna_path, index_path, t2g_path,
                nucleus=nucleus, k=k, include=include, exclude=exclude,
                threads=threads, dlist=dlist, dlist_overhang=dlist_overhang,
                aa=aa, overwrite=overwrite, make_unique=make_unique,
                temp_dir=temp_dir, max_ec_size=max_ec_size
            )
            
        print(f"{Colors.GREEN}✓ ref workflow completed!{Colors.ENDC}") 

        # Add extra information to the result
        if isinstance(result, dict):
            result.update({
                'workflow': workflow,
                'technology': 'N/A', # No technology concept in ref stage
                'parameters': {
                    'threads': threads,
                    'k': k,
                    'overwrite': overwrite,
                    'workflow_type': workflow
                }
            })

        return result

    except Exception as e:
        print(f"{Colors.FAIL}✗ ref workflow failed: {e}{Colors.ENDC}", file=sys.stderr) 
        raise RuntimeError(f"ref workflow execution failed: {e}")

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            remove_directory(temp_dir)


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
    Generate count matrix from single-cell FASTQ files - using internal API
    ... (docstring unchanged) ...
    """
    print(f"{Colors.BOLD}{Colors.HEADER}🚀 Starting count workflow: {workflow}{Colors.ENDC}") 
    print(f"{Colors.CYAN}    Technology: {technology}{Colors.ENDC}") 
    print(f"{Colors.CYAN}    Output directory: {output_path}{Colors.ENDC}")

    # Import kb_python modules
    modules = _import_kb_python_modules()
    _count, _count_nac, _count_velocity, _count_velocity_smartseq3 = modules['count']
    make_directory, remove_directory = modules['utils']

    # Ensure output and temporary directories exist
    make_directory(output_path)
    make_directory(temp_dir)

    # Convert string path to list
    if isinstance(fastq_paths, str):
        fastq_paths = [fastq_paths]

    # Process loom_names parameter - ensure it is a list
    if isinstance(loom_names, str):
        loom_names_list = [name.strip() for name in loom_names.split(',')]
    elif isinstance(loom_names, list):
        loom_names_list = loom_names
    else:
        loom_names_list = None
    try:
        if workflow == 'nac':
            print(f"{Colors.BLUE}    Executing NAC workflow{Colors.ENDC}")
            result = _count_nac(
                index_path, t2g_path, c1, c2,
                technology, output_path, fastq_paths, whitelist_path,
                replacement_path, tcc=tcc, mm=mm, filter='bustools' if filter_barcodes else None,
                filter_threshold=filter_threshold, threads=threads, memory=memory,
                overwrite=overwrite, loom=loom, loom_names=loom_names_list, h5ad=h5ad,
                cellranger=cellranger, report=report, inspect=not kwargs.get('no_inspect', False),
                temp_dir=temp_dir, fragment_l=fragment_l, fragment_s=fragment_s,
                paired=parity == 'paired', genomebam=genomebam, strand=strand,
                umi_gene=technology.upper() not in ('BULK', 'SMARTSEQ2'),
                em=em, by_name=gene_names, sum_matrices=kwargs.get('sum', 'none'),
                gtf_path=kwargs.get('gtf'), chromosomes_path=kwargs.get('chromosomes'),
                inleaved=inleaved, demultiplexed=technology.upper() in ('BULK', 'SMARTSEQ2'),
                batch_barcodes=batch_barcodes, numreads=numreads, store_num=store_num,
                lr=long_read, lr_thresh=threshold, lr_platform=platform,
                union=kwargs.get('union', False), no_jump=kwargs.get('no_jump', False),
                quant_umis=kwargs.get('quant_umis', False), keep_flags=kwargs.get('keep_flags', False),
                exact_barcodes=exact_barcodes
            )

        elif workflow in {'nucleus', 'lamanno'}:
            # 详情：使用 print，蓝色，并添加4个空格
            print(f"{Colors.BLUE}    Executing {workflow} workflow{Colors.ENDC}") 
            if technology.upper() == 'SMARTSEQ3':
                result = _count_velocity_smartseq3(
                    index_path, t2g_path, c1, c2,
                    output_path, fastq_paths, whitelist_path, tcc=tcc, mm=mm,
                    temp_dir=temp_dir, threads=threads, memory=memory,
                    overwrite=overwrite, loom=loom, h5ad=h5ad,
                    inspect=not kwargs.get('no_inspect', False),
                    strand=strand, by_name=gene_names
                )
            else:
                result = _count_velocity(
                    index_path, t2g_path, c1, c2,
                    technology, output_path, fastq_paths,
                    whitelist_path, tcc=tcc, mm=mm, filter='bustools' if filter_barcodes else None,
                    filter_threshold=filter_threshold, threads=threads, memory=memory,
                    overwrite=overwrite, loom=loom, h5ad=h5ad, cellranger=cellranger,
                    report=report, inspect=not kwargs.get('no_inspect', False),
                    nucleus=workflow == 'nucleus', temp_dir=temp_dir,
                    strand=strand, umi_gene=technology.upper() not in ('BULK', 'SMARTSEQ2'),
                    em=em, by_name=gene_names
                )

        else:
            # 详情：使用 print，蓝色，并添加4个空格
            print(f"{Colors.BLUE}    Executing standard workflow{Colors.ENDC}") # <--- 修改
            kite_workflow = 'kite' in workflow
            FB_workflow = '10xFB' in workflow

            result = _count(
                index_path, t2g_path, technology, output_path, fastq_paths,
                whitelist_path, replacement_path, tcc=tcc, mm=mm,
                filter='bustools' if filter_barcodes else None,
                filter_threshold=filter_threshold, kite=kite_workflow, FB=FB_workflow,
                threads=threads, memory=memory, overwrite=overwrite,
                loom=loom, loom_names=loom_names_list, h5ad=h5ad,
                cellranger=cellranger, report=report,
                inspect=not kwargs.get('no_inspect', False), temp_dir=temp_dir,
                fragment_l=fragment_l, fragment_s=fragment_s, paired=parity == 'paired',
                genomebam=genomebam, aa=aa, strand=strand,
                umi_gene=technology.upper() not in ('BULK', 'SMARTSEQ2'),
                em=em, by_name=gene_names, gtf_path=kwargs.get('gtf'),
                chromosomes_path=kwargs.get('chromosomes'), inleaved=inleaved,
                demultiplexed=technology.upper() in ('BULK', 'SMARTSEQ2'),
                batch_barcodes=batch_barcodes, bootstraps=bootstraps,
                matrix_to_files=kwargs.get('matrix_to_files', False),
                matrix_to_directories=kwargs.get('matrix_to_directories', False),
                no_fragment=kwargs.get('no_fragment', False),
                numreads=numreads, store_num=store_num, lr=long_read,
                lr_thresh=threshold, lr_error_rate=kwargs.get('error_rate'),
                lr_platform=platform, union=kwargs.get('union', False),
                no_jump=kwargs.get('no_jump', False),
                quant_umis=kwargs.get('quant_umis', False),
                keep_flags=kwargs.get('keep_flags', False),
                exact_barcodes=exact_barcodes
            )

        print(f"{Colors.GREEN}✓ count workflow completed!{Colors.ENDC}") # <--- 改为 print, Colors.GREEN, Colors.ENDC

        # Organize and return results
        output_info = {
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

        # Check generated files
        if h5ad:
            h5ad_file = os.path.join(output_path, "adata.h5ad")
            if os.path.exists(h5ad_file):
                output_info['h5ad_file'] = h5ad_file

        if loom:
            loom_file = os.path.join(output_path, "adata.loom")
            if os.path.exists(loom_file):
                output_info['loom_file'] = loom_file

        if cellranger:
            cellranger_dir = os.path.join(output_path, "cellranger")
            if os.path.exists(cellranger_dir):
                output_info['cellranger_dir'] = cellranger_dir

        # Standard matrix files
        matrix_file = os.path.join(output_path, "matrix.mtx")
        if os.path.exists(matrix_file):
            output_info['matrix_file'] = matrix_file

        barcodes_file = os.path.join(output_path, "barcodes.tsv")
        if os.path.exists(barcodes_file):
            output_info['barcodes_file'] = barcodes_file

        genes_file = os.path.join(output_path, "genes.tsv")
        if os.path.exists(genes_file):
            output_info['genes_file'] = genes_file

        # Merge results
        if isinstance(result, dict):
            output_info.update(result)

        return output_info

    except Exception as e:
        print(f"{Colors.FAIL}✗ count workflow failed: {e}{Colors.ENDC}", file=sys.stderr) # <--- 改为 print, Colors.FAIL, Colors.ENDC
        raise RuntimeError(f"count workflow execution failed: {e}")

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            remove_directory(temp_dir)

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
    One-click 10x v3 data analysis: Download reference + Count analysis

    Executes the complete analysis pipeline using the internal API
    """
    results = {}

    if download_reference:
        print(f"{Colors.BOLD}{Colors.HEADER}Step 1: Downloading human reference genome...{Colors.ENDC}")
        ref_result = ref(
            index_path=os.path.join(reference_output_dir, "index.idx"),
            t2g_path=os.path.join(reference_output_dir, "t2g.txt"),
            d='human',
            cdna_path=os.path.join(reference_output_dir, "transcriptome.fasta"),
            threads=threads_ref
        )
        results['reference'] = ref_result
        print(f"{Colors.GREEN}✓ Reference genome download complete!{Colors.ENDC}\n") 
        index_file = os.path.join(reference_output_dir, "index.idx")
        t2g_file = os.path.join(reference_output_dir, "t2g.txt")
    else:
        index_file = os.path.join(reference_output_dir, "index.idx")
        t2g_file = os.path.join(reference_output_dir, "t2g.txt")

    print(f"{Colors.BOLD}{Colors.HEADER}Step 2: Performing count analysis...{Colors.ENDC}") 
    count_result = count(
        fastq_paths=fastq_files,
        index_path=index_file,
        t2g_path=t2g_file,
        output_path=analysis_output_dir,
        technology='10XV3',
        threads=threads_count,
        **kwargs
    )
    results['count'] = count_result
    print(f"{Colors.GREEN}✓ Count analysis complete!{Colors.ENDC}\n") #

    return results
