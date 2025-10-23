import os
import re
import copy
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import scipy.io
from typing_extensions import Literal

from .config import get_bustools_binary_path, get_kallisto_binary_path
from .constants import (
    ABUNDANCE_FILENAME,
    ABUNDANCE_GENE_FILENAME,
    ABUNDANCE_GENE_TPM_FILENAME,
    ABUNDANCE_GENE_NAMES_FILENAME,
    ABUNDANCE_TPM_FILENAME,
    ADATA_PREFIX,
    BUS_FILENAME,
    CAPTURE_FILENAME,
    CELLRANGER_BARCODES,
    CELLRANGER_DIR,
    CELLRANGER_GENES,
    CELLRANGER_MATRIX,
    CORRECT_CODE,
    COUNTS_PREFIX,
    ECMAP_FILENAME,
    FEATURE_NAME,
    FEATURE_PREFIX,
    FILTER_WHITELIST_FILENAME,
    FILTERED_CODE,
    FILTERED_COUNTS_DIR,
    FLD_FILENAME,
    FLENS_FILENAME,
    GENE_NAME,
    GENES_FILENAME,
    GENE_NAMES_FILENAME,
    GENOMEBAM_FILENAME,
    GENOMEBAM_INDEX_FILENAME,
    INSPECT_FILENAME,
    INSPECT_INTERNAL_FILENAME,
    INSPECT_UMI_FILENAME,
    INTERNAL_SUFFIX,
    KALLISTO_INFO_FILENAME,
    KB_INFO_FILENAME,
    PROJECT_CODE,
    REPORT_HTML_FILENAME,
    REPORT_NOTEBOOK_FILENAME,
    SAVED_INDEX_FILENAME,
    SORT_CODE,
    TCC_PREFIX,
    TRANSCRIPT_NAME,
    TXNAMES_FILENAME,
    UMI_SUFFIX,
    UNFILTERED_CODE,
    UNFILTERED_COUNTS_DIR,
    UNFILTERED_QUANT_DIR,
    WHITELIST_FILENAME,
)
from .dry import dryable
from .dry import count as dry_count
from .logging import logger
from .report import render_report
from .utils import (
    copy_whitelist,
    create_10x_feature_barcode_map,
    get_temporary_filename,
    import_matrix_as_anndata,
    import_tcc_matrix_as_anndata,
    make_directory,
    open_as_text,
    overlay_anndatas,
    read_t2g,
    remove_directory,
    run_executable,
    stream_file,
    sum_anndatas,
    update_filename,
    whitelist_provided,
    obtain_gene_names,
    write_list_to_file,
    do_sum_matrices,
    move_file,
)
from .stats import STATS
from .validate import validate_files

INSPECT_PARSER = re.compile(r'^.*?(?P<count>[0-9]+)')


def make_transcript_t2g(txnames_path: str, out_path: str) -> str:
    """Make a two-column t2g file from a transcripts file

    Args:
        txnames_path: Path to transcripts.txt
        out_path: Path to output t2g file

    Returns:
       Path to output t2g file
    """
    with open_as_text(txnames_path, 'r') as f, open_as_text(out_path,
                                                            'w') as out:
        for line in f:
            out.write(f'{line.strip()}\t{line.strip()}\n')
    return out_path


def kallisto_bus(
    fastqs: Union[List[str], str],
    index_path: str,
    technology: str,
    out_dir: str,
    threads: int = 8,
    n: bool = False,
    k: bool = False,
    paired: bool = False,
    genomebam: bool = False,
    aa: bool = False,
    strand: Optional[Literal['unstranded', 'forward', 'reverse']] = None,
    gtf_path: Optional[str] = None,
    chromosomes_path: Optional[str] = None,
    inleaved: bool = False,
    demultiplexed: bool = False,
    batch_barcodes: bool = False,
    numreads: int = None,
    lr: bool = False,
    lr_thresh: float = 0.8,
    lr_error_rate: float = None,
    union: bool = False,
    no_jump: bool = False,
) -> Dict[str, str]:
    """Runs `kallisto bus`.

    Args:
        fastqs: List of FASTQ file paths, or a single path to a batch file
        index_path: Path to kallisto index
        technology: Single-cell technology used
        out_dir: Path to output directory
        threads: Number of threads to use, defaults to `8`
        n: Include number of read in flag column (used when splitting indices),
            defaults to `False`
        k: Alignment is done per k-mer (used when splitting indices),
            defaults to `False`
        paired: Whether or not to supply the `--paired` flag, only used for
            bulk and smartseq2 samples, defaults to `False`
        genomebam: Project pseudoalignments to genome sorted BAM file, defaults to
            `False`
        aa: Align to index generated from a FASTA-file containing amino acid sequences,
            defaults to `False`
        strand: Strandedness, defaults to `None`
        gtf_path: GTF file for transcriptome information (required for --genomebam),
            defaults to `None`
        chromosomes_path: Tab separated file with chromosome names and lengths
            (optional for --genomebam, but recommended), defaults to `None`
        inleaved: Whether input FASTQ is interleaved, defaults to `False`
        demultiplexed: Whether FASTQs are demultiplexed, defaults to `False`
        batch_barcodes: Whether sample ID should be in barcode, defaults to `False`
        numreads: Maximum number of reads to process from supplied input
        lr: Whether to use lr-kallisto in read mapping, defaults to `False`
        lr_thresh: Sets the --threshold for lr-kallisto, defaults to `0.8`
        lr_error_rate: Sets the --error-rate for lr-kallisto, defaults to `None`
        union: Use set union for pseudoalignment, defaults to `False`
        no_jump: Disable pseudoalignment "jumping", defaults to `False`

    Returns:
        Dictionary containing paths to generated files
    """
    logger.info(
        f'Using index {index_path} to generate BUS file to {out_dir} from'
    )
    results = {
        'bus': os.path.join(out_dir, BUS_FILENAME),
        'ecmap': os.path.join(out_dir, ECMAP_FILENAME),
        'txnames': os.path.join(out_dir, TXNAMES_FILENAME),
        'info': os.path.join(out_dir, KALLISTO_INFO_FILENAME)
    }
    is_batch = isinstance(fastqs, str)

    for fastq in [fastqs] if is_batch else fastqs:
        logger.info((' ' * 8) + fastq)
    command = [get_kallisto_binary_path(), 'bus']
    command += ['-i', index_path]
    command += ['-o', out_dir]
    if not demultiplexed:
        if technology.upper() == "10XV4":
            # TODO: REMOVE THIS WHEN KALLISTO IS UPDATED
            command += ['-x', "10XV3"]
        else:
            command += ['-x', technology]
    elif technology[0] == '-':
        # User supplied a custom demuxed (no-barcode) technology
        command += ['-x', technology]
    else:
        command += ['-x', 'BULK']
    command += ['-t', threads]
    if n:
        command += ['--num']
    if k:
        command += ['--kmer']
    if paired and not aa:
        command += ['--paired']
        results['flens'] = os.path.join(out_dir, FLENS_FILENAME)
    if genomebam:
        command += ['--genomebam']
        if gtf_path is not None:
            command += ['-g', gtf_path]
        if chromosomes_path is not None:
            command += ['-c', chromosomes_path]
        results['genomebam'] = os.path.join(out_dir, GENOMEBAM_FILENAME)
        results['genomebam_index'] = os.path.join(
            out_dir, GENOMEBAM_INDEX_FILENAME
        )
    if numreads:
        command += ['-N', numreads]
    if aa:
        command += ['--aa']
        if paired:
            logger.warning(
                '`--paired` ignored since `--aa` only supports single-end reads'
            )
    if strand == 'unstranded':
        command += ['--unstranded']
    elif strand == 'forward':
        command += ['--fr-stranded']
    elif strand == 'reverse':
        command += ['--rf-stranded']
    if inleaved:
        command += ['--inleaved']
    if lr:
        command += ['--long']
    if lr and lr_thresh:
        command += ['-r', str(lr_thresh)]
    if lr and lr_error_rate:
        command += ['-e', str(lr_error_rate)]
    if union:
        command += ['--union']
    if no_jump:
        command += ['--no-jump']
    if batch_barcodes:
        command += ['--batch-barcodes']
    if is_batch:
        command += ['--batch', fastqs]
    else:
        command += fastqs
    run_executable(command)

    if technology.upper() in ('BULK', 'SMARTSEQ3'):
        results['saved_index'] = os.path.join(out_dir, SAVED_INDEX_FILENAME)
        if os.path.exists(results['saved_index']):
            os.remove(results['saved_index'])  # TODO: Fix this in kallisto?
    return results


@validate_files(pre=False)
def kallisto_quant_tcc(
    mtx_path: str,
    saved_index_path: str,
    ecmap_path: str,
    t2g_path: str,
    out_dir: str,
    flens_path: Optional[str] = None,
    l: Optional[int] = None,
    s: Optional[int] = None,
    threads: int = 8,
    bootstraps: int = 0,
    matrix_to_files: bool = False,
    matrix_to_directories: bool = False,
    no_fragment: bool = False,
    lr: bool = False,
    lr_platform: str = 'ONT',
) -> Dict[str, str]:
    """Runs `kallisto quant-tcc`.

    Args:
        mtx_path: Path to counts matrix
        saved_index_path: Path to index
        ecmap_path: Path to ecmap
        t2g_path: Path to T2G
        out_dir: Output directory path
        flens_path: Path to flens.txt, defaults to `None`
        l: Mean fragment length, defaults to `None`
        s: Standard deviation of fragment length, defaults to `None`
        threads: Number of threads to use, defaults to `8`
        bootstraps: Number of bootstraps to perform for quant-tcc, defaults to 0
        matrix_to_files: Whether to write quant-tcc output to files, defaults to `False`
        matrix_to_directories: Whether to write quant-tcc output to directories, defaults to `False`
        no_fragment: Whether to disable quant-tcc effective length normalization, defaults to `False`
        lr: Whether to use lr-kallisto in quantification, defaults to `False`
        lr_platform: Sets the --platform for lr-kallisto, defaults to `ONT`

    Returns:
        Dictionary containing path to output files
    """
    logger.info(
        f'Quantifying transcript abundances to {out_dir} from mtx file {mtx_path}'
    )

    command = [get_kallisto_binary_path(), 'quant-tcc']
    command += ['-o', out_dir]
    command += ['-i', saved_index_path]
    command += ['-e', ecmap_path]
    command += ['-g', t2g_path]
    command += ['-t', threads]
    if lr:
        command += ['--long']
    if lr and lr_platform:
        command += ['-P', lr_platform]
    if flens_path and not no_fragment:
        command += ['-f', flens_path]
    if l and not no_fragment:
        command += ['-l', l]
    if s and not no_fragment:
        command += ['-s', s]
    if bootstraps and bootstraps != 0:
        command += ['-b', bootstraps]
    if matrix_to_files:
        command += ['--matrix-to-files']
    if matrix_to_directories:
        command += ['--matrix-to-directories']
    command += [mtx_path]
    run_executable(command)
    ret_dict = {
        'genes': os.path.join(out_dir, GENES_FILENAME),
        'gene_mtx': os.path.join(out_dir, ABUNDANCE_GENE_FILENAME),
        'gene_tpm_mtx': os.path.join(out_dir, ABUNDANCE_GENE_TPM_FILENAME),
        'mtx': os.path.join(out_dir, ABUNDANCE_FILENAME),
        'tpm_mtx': os.path.join(out_dir, ABUNDANCE_TPM_FILENAME),
        'txnames': os.path.join(out_dir, TXNAMES_FILENAME),
    }
    if flens_path or l or s:
        ret_dict['fld'] = os.path.join(out_dir, FLD_FILENAME)
    return ret_dict


@validate_files(pre=False)
def bustools_project(
    bus_path: str, out_path: str, map_path: str, ecmap_path: str,
    txnames_path: str
) -> Dict[str, str]:
    """Runs `bustools project`.

        bus_path: Path to BUS file to sort
        out_dir: Path to output directory
        map_path: Path to file containing source-to-destination mapping
        ecmap_path: Path to ecmap file, as generated by `kallisto bus`
        txnames_path: Path to transcript names file, as generated by `kallisto bus`

    Returns:
        Dictionary containing path to generated BUS file
    """
    logger.info('Projecting BUS file {} with map {}'.format(bus_path, map_path))
    command = [get_bustools_binary_path(), 'project']
    command += ['-o', out_path]
    command += ['-m', map_path]
    command += ['-e', ecmap_path]
    command += ['-t', txnames_path]
    command += ['--barcode']
    command += [bus_path]
    run_executable(command)
    return {'bus': out_path}


def bustools_sort(
    bus_path: str,
    out_path: str,
    temp_dir: str = 'tmp',
    threads: int = 8,
    memory: str = '2G',
    flags: bool = False,
    store_num: bool = False,
) -> Dict[str, str]:
    """Runs `bustools sort`.

    Args:
        bus_path: Path to BUS file to sort
        out_dir: Path to output BUS path
        temp_dir: Path to temporary directory, defaults to `tmp`
        threads: Number of threads to use, defaults to `8`
        memory: Amount of memory to use, defaults to `2G`
        flags: Whether to supply the `--flags` argument to sort, defaults to
            `False`
        store_num: Whether to process BUS files with read numbers in flag,
            defaults to `False`

    Returns:
        Dictionary containing path to generated index
    """
    logger.info('Sorting BUS file {} to {}'.format(bus_path, out_path))
    command = [get_bustools_binary_path(), 'sort']
    command += ['-o', out_path]
    command += ['-T', temp_dir]
    command += ['-t', threads]
    command += ['-m', memory]
    if flags:
        command += ['--flags']
    if store_num:
        command += ['--no-flags']
    command += [bus_path]
    run_executable(command)
    return {'bus': out_path}


@validate_files(pre=False)
def bustools_inspect(
    bus_path: str,
    out_path: str,
    whitelist_path: Optional[str] = None,
    ecmap_path: Optional[str] = None,
) -> Dict[str, str]:
    """Runs `bustools inspect`.

    Args:
        bus_path: Path to BUS file to sort
        out_path: Path to output inspect JSON file
        whitelist_path: Path to whitelist
        ecmap_path: Path to ecmap file, as generated by `kallisto bus`

    Returns:
        Dictionary containing path to generated index
    """
    logger.info('Inspecting BUS file {}'.format(bus_path))
    command = [get_bustools_binary_path(), 'inspect']
    command += ['-o', out_path]
    if whitelist_path:
        command += ['-w', whitelist_path]
    if ecmap_path:
        command += ['-e', ecmap_path]
    command += [bus_path]
    run_executable(command)
    return {'inspect': out_path}


def bustools_correct(
    bus_path: str,
    out_path: str,
    whitelist_path: str,
    replace: bool = False,
    exact_barcodes: bool = False
) -> Dict[str, str]:
    """Runs `bustools correct`.

    Args:
        bus_path: Path to BUS file to correct
        out_path: Path to output corrected BUS file
        whitelist_path: Path to whitelist
        replace: If whitelist is a replacement file, defaults to `False`
        exact_barcodes: Use exact matching for 'correction', defaults to `False`

    Returns:
        Dictionary containing path to generated index
    """
    logger.info(
        'Correcting BUS records in {} to {} with on-list {}'.format(
            bus_path, out_path, whitelist_path
        )
    )
    command = [get_bustools_binary_path(), 'correct']
    command += ['-o', out_path]
    command += ['-w', whitelist_path]
    command += [bus_path]
    if replace:
        command += ['--replace']
    if exact_barcodes:
        command += ['--nocorrect']
    run_executable(command)
    return {'bus': out_path}


@validate_files(pre=False)
def bustools_count(
    bus_path: str,
    out_prefix: str,
    t2g_path: str,
    ecmap_path: str,
    txnames_path: str,
    tcc: bool = False,
    mm: bool = False,
    cm: bool = False,
    umi_gene: bool = True,
    em: bool = False,
    nascent_path: str = None,
    batch_barcodes: bool = False,
) -> Dict[str, str]:
    """Runs `bustools count`.

    Args:
        bus_path: Path to BUS file to correct
        out_prefix: Prefix of the output files to generate
        t2g_path: Path to output transcript-to-gene mapping
        ecmap_path: Path to ecmap file, as generated by `kallisto bus`
        txnames_path: Path to transcript names file, as generated by `kallisto bus`
        tcc: Whether to generate a TCC matrix instead of a gene count matrix,
            defaults to `False`
        mm: Whether to include BUS records that pseudoalign to multiple genes,
            defaults to `False`
        cm: Count multiplicities instead of UMIs. Used for chemitries
            without UMIs, such as bulk and Smartseq2, defaults to `False`
        umi_gene: Whether to use genes to deduplicate umis, defaults to `True`
        em: Whether to estimate gene abundances using EM algorithm, defaults
            to `False`
        nascent_path: Path to list of nascent targets for obtaining
            nascent/mature/ambiguous matrices, defaults to `None`
        batch_barcodes: If sample ID is barcoded, defaults to `False`

    Returns:
        Dictionary containing path to generated index
    """
    logger.info(
        f'Generating count matrix {out_prefix} from BUS file {bus_path}'
    )
    command = [get_bustools_binary_path(), 'count']
    command += ['-o', out_prefix]
    command += ['-g', t2g_path]
    command += ['-e', ecmap_path]
    command += ['-t', txnames_path]
    if nascent_path:
        command += ['-s', nascent_path]
    if not tcc:
        command += ['--genecounts']
    if mm:
        command += ['--multimapping']
    if cm:
        command += ['--cm']
    if umi_gene and not cm:
        command += ['--umi-gene']
    if em:
        command += ['--em']
    command += [bus_path]

    # There is currently a bug when a directory with the same path as `out_prefix`
    # exists, the matrix is named incorrectly. So, to get around this, manually
    # detect and remove such a directory should it exist.
    if os.path.isdir(out_prefix):
        remove_directory(out_prefix)

    run_executable(command)
    if nascent_path:
        ret = {
            'mtx0':
                move_file(f'{out_prefix}.mtx', f'{out_prefix}.mature.mtx'),
            'ec0' if tcc else 'genes0':
                f'{out_prefix}.ec.txt' if tcc else f'{out_prefix}.genes.txt',
            'barcodes0':
                f'{out_prefix}.barcodes.txt',
            'batch_barcodes0':
                f'{out_prefix}.barcodes.prefix.txt' if batch_barcodes else None,
            'mtx1':
                move_file(f'{out_prefix}.2.mtx', f'{out_prefix}.nascent.mtx'),
            'ec1' if tcc else 'genes1':
                f'{out_prefix}.ec.txt' if tcc else f'{out_prefix}.genes.txt',
            'barcodes1':
                f'{out_prefix}.barcodes.txt',
            'batch_barcodes1':
                f'{out_prefix}.barcodes.prefix.txt' if batch_barcodes else None,
            'mtx2':
                f'{out_prefix}.ambiguous.mtx',
            'ec2' if tcc else 'genes2':
                f'{out_prefix}.ec.txt' if tcc else f'{out_prefix}.genes.txt',
            'barcodes2':
                f'{out_prefix}.barcodes.txt',
            'batch_barcodes2':
                f'{out_prefix}.barcodes.prefix.txt' if batch_barcodes else None,
        }
        if not batch_barcodes:
            del ret['batch_barcodes0']
            del ret['batch_barcodes1']
            del ret['batch_barcodes2']
        elif not os.path.exists(ret['batch_barcodes0']):
            del ret['batch_barcodes0']
            del ret['batch_barcodes1']
            del ret['batch_barcodes2']
        return ret
    ret = {
        'mtx':
            f'{out_prefix}.mtx',
        'ec' if tcc else 'genes':
            f'{out_prefix}.ec.txt' if tcc else f'{out_prefix}.genes.txt',
        'barcodes':
            f'{out_prefix}.barcodes.txt',
        'batch_barcodes':
            f'{out_prefix}.barcodes.prefix.txt' if batch_barcodes else None,
    }
    if not batch_barcodes:
        del ret['batch_barcodes']
    elif not os.path.exists(ret['batch_barcodes']):
        del ret['batch_barcodes']
    return ret


@validate_files(pre=False)
def bustools_capture(
    bus_path: str,
    out_path: str,
    capture_path: str,
    ecmap_path: Optional[str] = None,
    txnames_path: Optional[str] = None,
    capture_type: Literal['transcripts', 'umis', 'barcode'] = 'transcripts',
    complement: bool = True,
) -> Dict[str, str]:
    """Runs `bustools capture`.

    Args:
        bus_path: Path to BUS file to capture
        out_path: Path to BUS file to generate
        capture_path: Path transcripts-to-capture list
        ecmap_path: Path to ecmap file, as generated by `kallisto bus`
        txnames_path: Path to transcript names file, as generated by `kallisto bus`
        capture_type: The type of information in the capture list. Can be one of
            `transcripts`, `umis`, `barcode`.
        complement: Whether or not to complement, defaults to `True`

    Returns:
        Dictionary containing path to generated index
    """
    logger.info(
        f'Capturing records from BUS file {bus_path} to {out_path} with capture list {capture_path}'
    )
    command = [get_bustools_binary_path(), 'capture']
    command += ['-o', out_path]
    command += ['-c', capture_path]
    if ecmap_path:
        command += ['-e', ecmap_path]
    if txnames_path:
        command += ['-t', txnames_path]
    if complement:
        command += ['--complement']
    command += ['--{}'.format(capture_type)]
    command += [bus_path]
    run_executable(command)
    return {'bus': out_path}


@validate_files(pre=False)
def bustools_whitelist(
    bus_path: str,
    out_path: str,
    threshold: Optional[int] = None
) -> Dict[str, str]:
    """Runs `bustools allowlist`.

    Args:
        bus_path: Path to BUS file generate the on-list from
        out_path: Path to output on-list
        threshold: Barcode threshold to be included in on-list

    Returns:
        Dictionary containing path to generated index
    """
    logger.info(
        'Generating on-list {} from BUS file {}'.format(out_path, bus_path)
    )
    command = [get_bustools_binary_path(), 'allowlist']
    command += ['-o', out_path]
    if threshold:
        command += ['--threshold', threshold]
    command += [bus_path]
    run_executable(command)
    return {'whitelist': out_path}


def matrix_to_cellranger(
    matrix_path: str, barcodes_path: str, genes_path: str, t2g_path: str,
    out_dir: str
) -> Dict[str, str]:
    """Convert bustools count matrix to cellranger-format matrix.

    Args:
        matrix_path: Path to matrix
        barcodes_path: List of paths to barcodes.txt
        genes_path: Path to genes.txt
        t2g_path: Path to transcript-to-gene mapping
        out_dir: Path to output matrix

    Returns:
        Dictionary of matrix files
    """
    make_directory(out_dir)
    logger.info(f'Writing matrix in cellranger format to {out_dir}')

    cr_matrix_path = os.path.join(out_dir, CELLRANGER_MATRIX)
    cr_barcodes_path = os.path.join(out_dir, CELLRANGER_BARCODES)
    cr_genes_path = os.path.join(out_dir, CELLRANGER_GENES)

    # Cellranger outputs genes x cells matrix
    mtx = scipy.io.mmread(matrix_path)
    scipy.io.mmwrite(cr_matrix_path, mtx.T, field='integer')

    with open(barcodes_path, 'r') as f, open(cr_barcodes_path, 'w') as out:
        for line in f:
            if line.isspace():
                continue
            out.write(f'{line.strip()}-1\n')

    # Get all (available) gene names
    gene_to_name = {}
    with open(t2g_path, 'r') as f:
        for line in f:
            if line.isspace():
                continue
            split = line.strip().split('\t')
            if len(split) > 2:
                gene_to_name[split[1]] = split[2]

    with open(genes_path, 'r') as f, open(cr_genes_path, 'w') as out:
        for line in f:
            if line.isspace():
                continue
            gene = line.strip()
            gene_name = gene_to_name.get(gene, gene)
            out.write(f'{gene}\t{gene_name}\n')

    return {
        'mtx': cr_matrix_path,
        'barcodes': cr_barcodes_path,
        'genes': cr_genes_path
    }


def convert_matrix(
    counts_dir: str,
    matrix_path: str,
    barcodes_path: str,
    batch_barcodes_path: Optional[str] = None,
    genes_path: Optional[str] = None,
    ec_path: Optional[str] = None,
    t2g_path: Optional[str] = None,
    txnames_path: Optional[str] = None,
    name: str = 'gene',
    loom: bool = False,
    loom_names: List[str] = ['barcode', 'target_name'],
    h5ad: bool = False,
    by_name: bool = False,
    tcc: bool = False,
    threads: int = 8,
) -> Dict[str, str]:
    """Convert a gene count or TCC matrix to loom or h5ad.

    Args:
        counts_dir: Path to counts directory
        matrix_path: Path to matrix
        barcodes_path: List of paths to barcodes.txt
        batch_barcodes_path: Path to barcodes prefixed with sample ID,
            defaults to `None`
        genes_path: Path to genes.txt, defaults to `None`
        ec_path: Path to ec.txt, defaults to `None`
        t2g_path: Path to transcript-to-gene mapping. If this is provided,
            the third column of the mapping is appended to the anndata var,
            defaults to `None`
        txnames_path: Path to transcripts.txt, defaults to `None`
        name: Name of the columns, defaults to "gene"
        loom: Whether to generate loom file, defaults to `False`
        loom_names: Names for col_attrs and row_attrs in loom file,
            defaults to `['barcode','target_name']`
        h5ad: Whether to generate h5ad file, defaults to `False`
        by_name: Aggregate counts by name instead of ID.
        tcc: Whether the matrix is a TCC matrix, defaults to `False`
        threads: Number of threads to use, defaults to `8`

    Returns:
        Dictionary of generated files
    """
    results = {}
    logger.info(f'Reading matrix {matrix_path}')
    adata = import_tcc_matrix_as_anndata(
        matrix_path,
        barcodes_path,
        ec_path,
        txnames_path,
        threads=threads,
        loom=loom,
        loom_names=loom_names,
        batch_barcodes_path=batch_barcodes_path
    ) if tcc else import_matrix_as_anndata(
        matrix_path,
        barcodes_path,
        genes_path,
        t2g_path=t2g_path,
        name=name,
        by_name=by_name,
        loom=loom,
        loom_names=loom_names,
        batch_barcodes_path=batch_barcodes_path
    )
    if loom:
        loom_path = os.path.join(counts_dir, f'{ADATA_PREFIX}.loom')
        logger.info(f'Writing matrix to loom {loom_path}')
        adata.write_loom(loom_path)
        results.update({'loom': loom_path})
    if h5ad:
        h5ad_path = os.path.join(counts_dir, f'{ADATA_PREFIX}.h5ad')
        logger.info(f'Writing matrix to h5ad {h5ad_path}')
        adata.write(h5ad_path)
        results.update({'h5ad': h5ad_path})

    return results


def convert_matrices(
    counts_dir: str,
    matrix_paths: List[str],
    barcodes_paths: List[str],
    batch_barcodes_paths: Optional[List[str]] = None,
    genes_paths: Optional[List[str]] = None,
    ec_paths: Optional[List[str]] = None,
    t2g_path: Optional[str] = None,
    txnames_path: Optional[str] = None,
    name: str = 'gene',
    loom: bool = False,
    loom_names: List[str] = ['barcode', 'target_name'],
    h5ad: bool = False,
    by_name: bool = False,
    nucleus: bool = False,
    tcc: bool = False,
    threads: int = 8,
) -> Dict[str, str]:
    """Convert a gene count or TCC matrix to loom or h5ad.

    Args:
        counts_dir: Path to counts directory
        matrix_paths: List of paths to matrices
        barcodes_paths: List of paths to barcodes.txt
        batch_barcodes_path: Paths to barcodes prefixed with sample ID,
            defaults to `None`
        genes_paths: List of paths to genes.txt, defaults to `None`
        ec_paths: List of path to ec.txt, defaults to `None`
        t2g_path: Path to transcript-to-gene mapping. If this is provided,
            the third column of the mapping is appended to the anndata var,
            defaults to `None`
        txnames_path: List of paths to transcripts.txt, defaults to `None`
        name: Name of the columns, defaults to "gene"
        loom: Whether to generate loom file, defaults to `False`
        loom_names: Names for col_attrs and row_attrs in loom file,
            defaults to `['barcode','target_name']`
        h5ad: Whether to generate h5ad file, defaults to `False`
        by_name: Aggregate counts by name instead of ID.
        nucleus: Whether the matrices contain single nucleus counts, defaults to `False`
        tcc: Whether the matrix is a TCC matrix, defaults to `False`
        threads: Number of threads to use, defaults to `8`

    Returns:
        Dictionary of generated files
    """
    results = {}
    adatas = []
    matrix_paths = matrix_paths or []
    barcodes_paths = barcodes_paths or []
    batch_barcodes_paths = batch_barcodes_paths or []
    if not batch_barcodes_paths:
        batch_barcodes_paths = [None for x in matrix_paths]
    genes_paths = genes_paths or []
    ec_paths = ec_paths or []
    for matrix_path, barcodes_path, batch_barcodes_path, genes_ec_path in zip(
            matrix_paths, barcodes_paths, batch_barcodes_paths, ec_paths
            if not genes_paths or None in genes_paths else genes_paths):
        logger.info(f'Reading matrix {matrix_path}')
        adatas.append(
            import_tcc_matrix_as_anndata(
                matrix_path,
                barcodes_path,
                genes_ec_path,
                txnames_path,
                threads=threads,
                loom=loom,
                loom_names=loom_names,
                batch_barcodes_path=batch_barcodes_path
            ) if tcc else import_matrix_as_anndata(
                matrix_path,
                barcodes_path,
                genes_ec_path,
                t2g_path=t2g_path,
                name=name,
                by_name=by_name,
                loom=loom,
                loom_names=loom_names,
                batch_barcodes_path=batch_barcodes_path
            )
        )
    logger.info('Combining matrices')
    adata = sum_anndatas(*adatas) if nucleus else overlay_anndatas(*adatas)
    if loom:
        loom_path = os.path.join(counts_dir, f'{ADATA_PREFIX}.loom')
        logger.info(f'Writing matrices to loom {loom_path}')
        adata.write_loom(loom_path)
        results.update({'loom': loom_path})
    if h5ad:
        h5ad_path = os.path.join(counts_dir, f'{ADATA_PREFIX}.h5ad')
        logger.info(f'Writing matrices to h5ad {h5ad_path}')
        adata.write(h5ad_path)
        results.update({'h5ad': h5ad_path})
    return results


def count_result_to_dict(count_result: Dict[str, str]) -> List[Dict[str, str]]:
    """Converts count result dict to list.

    Args:
        count_result: Count result object returned by bustools_count

    Returns:
        List of count result dicts
    """

    new_count_result = []
    for i in range(len(count_result)):
        if f'mtx{i}' not in count_result:
            break
        new_count_result.append({
            'mtx':
                count_result[f'mtx{i}'],
            'ec' if f'ec{i}' in count_result else 'genes':
                count_result[f'ec{i}' if f'ec{i}' in
                             count_result else f'genes{i}'],
            'barcodes':
                count_result[f'barcodes{i}'],
            'batch_barcodes':
                count_result[f'batch_barcodes{i}']
                if f'batch_barcodes{i}' in count_result else None,
        })
    return new_count_result


def filter_with_bustools(
    bus_path: str,
    ecmap_path: str,
    txnames_path: str,
    t2g_path: str,
    whitelist_path: str,
    filtered_bus_path: str,
    filter_threshold: Optional[int] = None,
    counts_prefix: Optional[str] = None,
    tcc: bool = False,
    mm: bool = False,
    kite: bool = False,
    temp_dir: str = 'tmp',
    threads: int = 8,
    memory: str = '2G',
    count: bool = True,
    loom: bool = False,
    loom_names: List[str] = ['barcode', 'target_name'],
    h5ad: bool = False,
    by_name: bool = False,
    cellranger: bool = False,
    umi_gene: bool = True,
    em: bool = False,
) -> Dict[str, str]:
    """Generate filtered count matrices with bustools.

    Args:
        bus_path: Path to sorted, corrected, sorted BUS file
        ecmap_path: Path to matrix ec file
        txnames_path: Path to list of transcripts
        t2g_path: Path to transcript-to-gene mapping
        whitelist_path: Path to filter whitelist to generate
        filtered_bus_path: Path to filtered BUS file to generate
        filter_threshold: Barcode filter threshold for bustools, defaults
            to `None`
        counts_prefix: Prefix of count matrix, defaults to `None`
        tcc: Whether to generate a TCC matrix instead of a gene count matrix,
            defaults to `False`
        mm: Whether to include BUS records that pseudoalign to multiple genes,
            defaults to `False`
        kite: Whether this is a KITE workflow
        temp_dir: Path to temporary directory, defaults to `tmp`
        threads: Number of threads to use, defaults to `8`
        memory: Amount of memory to use, defaults to `2G`
        count: Whether to run `bustools count`, defaults to `True`
        loom: Whether to convert the final count matrix into a loom file,
            defaults to `False`
        loom_names: Names for col_attrs and row_attrs in loom file,
            defaults to `['barcode','target_name']`
        h5ad: Whether to convert the final count matrix into a h5ad file,
            defaults to `False`
        by_name: Aggregate counts by name instead of ID.
        cellranger: Whether to convert the final count matrix into a
            cellranger-compatible matrix, defaults to `False`
        umi_gene: Whether to perform gene-level UMI collapsing, defaults to
            `True`
        em: Whether to estimate gene abundances using EM algorithm, defaults to
            `False`

    Returns:
        Dictionary of generated files
    """
    logger.info('Filtering with bustools')
    results = {}
    whitelist_result = bustools_whitelist(
        bus_path, whitelist_path, threshold=filter_threshold
    )
    results.update(whitelist_result)
    correct_result = bustools_correct(
        bus_path,
        os.path.join(
            temp_dir, update_filename(os.path.basename(bus_path), CORRECT_CODE)
        ),
        whitelist_result['whitelist'],
    )
    sort_result = bustools_sort(
        correct_result['bus'],
        filtered_bus_path,
        temp_dir=temp_dir,
        threads=threads,
        memory=memory,
    )
    results.update({'bus_scs': sort_result['bus']})

    if count:
        counts_dir = os.path.dirname(counts_prefix)
        make_directory(counts_dir)
        count_result = bustools_count(
            sort_result['bus'],
            counts_prefix,
            t2g_path,
            ecmap_path,
            txnames_path,
            tcc=tcc,
            mm=mm,
            umi_gene=umi_gene,
            em=em,
        )
        results.update(count_result)

        if 'genes' in count_result:
            genes_by_name_path = f'{counts_prefix}.{GENE_NAMES_FILENAME}'
            logger.info(f'Writing gene names to file {genes_by_name_path}')
            genes_by_name = obtain_gene_names(
                t2g_path, count_result.get('genes')
            )
            if genes_by_name:
                results.update({
                    'genenames':
                        write_list_to_file(genes_by_name, genes_by_name_path)
                })
        if loom or h5ad:
            results.update(
                convert_matrix(
                    counts_dir,
                    count_result['mtx'],
                    count_result['barcodes'],
                    batch_barcodes_path=count_result['batch_barcodes']
                    if 'batch_barcodes' in count_result else None,
                    genes_path=count_result.get('genes'),
                    t2g_path=t2g_path,
                    ec_path=count_result.get('ec'),
                    txnames_path=txnames_path,
                    name=FEATURE_NAME if kite else GENE_NAME,
                    loom=loom,
                    loom_names=loom_names,
                    h5ad=h5ad,
                    by_name=by_name,
                    tcc=tcc,
                    threads=threads,
                )
            )
        if cellranger:
            if not tcc:
                cr_result = matrix_to_cellranger(
                    count_result['mtx'], count_result['barcodes'],
                    count_result['genes'], t2g_path,
                    os.path.join(counts_dir, CELLRANGER_DIR)
                )
                results.update({'cellranger': cr_result})
            else:
                logger.warning(
                    'TCC matrices can not be converted to cellranger-compatible format.'
                )

    return results


def stream_fastqs(fastqs: List[str], temp_dir: str = 'tmp') -> List[str]:
    """Given a list of fastqs (that may be local or remote paths), stream any
    remote files. Internally, calls utils.

    Args:
        fastqs: List of (remote or local) fastq paths
        temp_dir: Temporary directory

    Returns:
        All remote paths substituted with a local path
    """
    return [
        stream_file(fastq, os.path.join(temp_dir, os.path.basename(fastq)))
        if urlparse(fastq).scheme in ('http', 'https', 'ftp', 'ftps') else fastq
        for fastq in fastqs
    ]


@dryable(dry_count.stream_batch)
def stream_batch(batch_path: str, temp_dir: str = 'tmp') -> str:
    """Given a path to a batch file, produce a new batch file where all the
    remote FASTQs are being streamed.

    Args:
        fastqs: List of (remote or local) fastq paths
        temp_dir: Temporary directory

    Returns:
        New batch file with all remote paths substituted with a local path
    """
    new_batch_path = get_temporary_filename(temp_dir)
    with open(batch_path, 'r') as f_in, open(new_batch_path, 'w') as f_out:
        for line in f_in:
            if line.isspace() or line.startswith('#'):
                continue
            sep = '\t' if '\t' in line else ' '
            split = line.strip().split(sep)
            name = split[0]
            fastqs = stream_fastqs(split[1:])
            f_out.write(f'{name}\t' + '\t'.join(fastqs) + '\n')
    return new_batch_path


def copy_or_create_whitelist(
    technology: str, bus_path: str, out_dir: str
) -> str:
    """Copies a pre-packaged whitelist if it is provided. Otherwise, runs
    `bustools whitelist` to generate a whitelist.

    Args:
        technology: Single-cell technology used
        bus_path: Path to BUS file generate the whitelist from
        out_dir: Path to output directory

    Returns:
        Path to copied or generated whitelist
    """
    if whitelist_provided(technology):
        logger.info(
            'Copying pre-packaged {} on-list to {}'.format(
                technology.upper(), out_dir
            )
        )
        return copy_whitelist(technology, out_dir)
    else:
        return bustools_whitelist(
            bus_path, os.path.join(out_dir, WHITELIST_FILENAME)
        )['whitelist']


def convert_transcripts_to_genes(
    txnames_path: str, t2g_path: str, genes_path: str
) -> str:
    """Convert a textfile containing transcript IDs to another textfile containing
    gene IDs, given a transcript-to-gene mapping.

    Args:
        txnames_path: Path to transcripts.txt
        t2g_path: Path to transcript-to-genes mapping
        genes_path: Path to output genes.txt

    Returns:
        Path to written genes.txt
    """
    t2g = read_t2g(t2g_path)
    with open_as_text(txnames_path, 'r') as f, open_as_text(genes_path,
                                                            'w') as out:
        for line in f:
            if line.isspace():
                continue
            transcript = line.strip()
            if transcript not in t2g:
                logger.warning(
                    f'Transcript {transcript} was found in {txnames_path} but not in {t2g_path}. '
                    'This transcript will not be converted to a gene.'
                )
            attributes = t2g.get(transcript)

            if attributes:
                out.write(f'{attributes[0]}\n')
            else:
                out.write(f'{transcript}\n')
    return genes_path


@dryable(dry_count.write_smartseq3_capture)
def write_smartseq3_capture(capture_path: str) -> str:
    """Write the capture sequence for smartseq3.

    Args:
        capture_path: Path to write the capture sequence

    Returns:
        Path to written file
    """
    with open(capture_path, 'w') as f:
        f.write(('T' * 32) + '\n')
    return capture_path


@logger.namespaced('count')
def count(
    index_path: str,
    t2g_path: str,
    technology: str,
    out_dir: str,
    fastqs: List[str],
    whitelist_path: Optional[str] = None,
    replacement_path: Optional[str] = None,
    tcc: bool = False,
    mm: bool = False,
    filter: Optional[Literal['bustools']] = None,
    filter_threshold: Optional[int] = None,
    kite: bool = False,
    FB: bool = False,
    temp_dir: str = 'tmp',
    threads: int = 8,
    memory: str = '2G',
    overwrite: bool = False,
    loom: bool = False,
    loom_names: List[str] = ['barcode', 'target_name'],
    h5ad: bool = False,
    by_name: bool = False,
    cellranger: bool = False,
    inspect: bool = True,
    report: bool = False,
    fragment_l: Optional[int] = None,
    fragment_s: Optional[int] = None,
    paired: bool = False,
    genomebam: bool = False,
    aa: bool = False,
    strand: Optional[Literal['unstranded', 'forward', 'reverse']] = None,
    umi_gene: bool = True,
    em: bool = False,
    gtf_path: Optional[str] = None,
    chromosomes_path: Optional[str] = None,
    inleaved: bool = False,
    demultiplexed: bool = False,
    batch_barcodes: bool = False,
    bootstraps: int = 0,
    matrix_to_files: bool = False,
    matrix_to_directories: bool = False,
    no_fragment: bool = False,
    numreads: int = None,
    store_num: bool = False,
    lr: bool = False,
    lr_thresh: float = 0.8,
    lr_error_rate: float = None,
    lr_platform: str = 'ONT',
    union: bool = False,
    no_jump: bool = False,
    quant_umis: bool = False,
    keep_flags: bool = False,
    exact_barcodes: bool = False,
) -> Dict[str, Union[str, Dict[str, str]]]:
    """Generates count matrices for single-cell RNA seq.

    Args:
        index_path: Path to kallisto index
        t2g_path: Path to transcript-to-gene mapping
        technology: Single-cell technology used
        out_dir: Path to output directory
        fastqs: List of FASTQ file paths or a single batch definition file
        whitelist_path: Path to whitelist, defaults to `None`
        replacement_path: Path to replacement list, defaults to `None`
        tcc: Whether to generate a TCC matrix instead of a gene count matrix,
            defaults to `False`
        mm: Whether to include BUS records that pseudoalign to multiple genes,
            defaults to `False`
        filter: Filter to use to generate a filtered count matrix,
            defaults to `None`
        filter_threshold: Barcode filter threshold for bustools, defaults
            to `None`
        kite: Whether this is a KITE workflow
        FB: Whether 10x Genomics Feature Barcoding technology was used,
            defaults to `False`
        temp_dir: Path to temporary directory, defaults to `tmp`
        threads: Pumber of threads to use, defaults to `8`
        memory: Amount of memory to use, defaults to `2G`
        overwrite: Overwrite an existing index file, defaults to `False`
        loom: Whether to convert the final count matrix into a loom file,
            defaults to `False`
        loom_names: Names for col_attrs and row_attrs in loom file,
            defaults to `['barcode','target_name']`
        h5ad: Whether to convert the final count matrix into a h5ad file,
            defaults to `False`
        by_name: Aggregate counts by name instead of ID.
        cellranger: Whether to convert the final count matrix into a
            cellranger-compatible matrix, defaults to `False`
        inspect: Whether or not to inspect the output BUS file and generate
            the inspect.json
        report: Generate an HTMl report, defaults to `False`
        fragment_l: Mean length of fragments, defaults to `None`
        fragment_s: Standard deviation of fragment lengths, defaults to `None`
        paired: Whether the fastqs are paired. Has no effect when a single
            batch file is provided. Defaults to `False`
        genomebam: Project pseudoalignments to genome sorted BAM file, defaults to
            `False`
        aa: Align to index generated from a FASTA-file containing amino acid sequences,
            defaults to `False`
        strand: Strandedness, defaults to `None`
        umi_gene: Whether to perform gene-level UMI collapsing, defaults to
            `True`
        em: Whether to estimate gene abundances using EM algorithm,
            defaults to `False`
        gtf_path: GTF file for transcriptome information (required for --genomebam),
            defaults to `None`
        chromosomes_path: Tab separated file with chromosome names and lengths
            (optional for --genomebam, but recommended), defaults to `None`
        inleaved: Whether input FASTQ is interleaved, defaults to `False`
        demultiplexed: Whether FASTQs are demultiplexed, defaults to `False`
        batch_barcodes: Whether sample ID should be in barcode, defaults to `False`
        bootstraps: Number of bootstraps to perform for quant-tcc, defaults to 0
        matrix_to_files: Whether to write quant-tcc output to files, defaults to `False`
        matrix_to_directories: Whether to write quant-tcc output to directories, defaults to `False`
        no_fragment: Whether to disable quant-tcc effective length normalization, defaults to `False`
        numreads: Maximum number of reads to process from supplied input
        store_num: Whether to store read numbers in BUS file, defaults to `False`
        lr: Whether to use lr-kallisto in read mapping, defaults to `False`
        lr_thresh: Sets the --threshold for lr-kallisto, defaults to `0.8`
        lr_error_rate: Sets the --error-rate for lr-kallisto, defaults to `None`
        lr_platform: Sets the --platform for lr-kallisto, defaults to `ONT`
        union: Use set union for pseudoalignment, defaults to `False`
        no_jump: Disable pseudoalignment "jumping", defaults to `False`
        quant_umis: Whether to run quant-tcc when there are UMIs, defaults to `False`
        keep_flags: Preserve flag column when sorting BUS file, defaults to `False`
        exact_barcodes: Use exact match for 'correcting' barcodes to on-list, defaults to `False`

    Returns:
        Dictionary containing paths to generated files
    """
    STATS.start()
    is_batch = isinstance(fastqs, str)

    results = {}
    make_directory(out_dir)
    unfiltered_results = results.setdefault('unfiltered', {})

    bus_result = {
        'bus': os.path.join(out_dir, BUS_FILENAME),
        'ecmap': os.path.join(out_dir, ECMAP_FILENAME),
        'txnames': os.path.join(out_dir, TXNAMES_FILENAME),
        'info': os.path.join(out_dir, KALLISTO_INFO_FILENAME)
    }
    if technology.upper() in ('BULK', 'SMARTSEQ2', 'SMARTSEQ3'):
        bus_result['saved_index'] = os.path.join(out_dir, SAVED_INDEX_FILENAME)
        if technology.upper() == 'SMARTSEQ3':
            paired = True
    if paired:
        bus_result['flens'] = os.path.join(out_dir, FLENS_FILENAME)
    if any(not os.path.exists(path)
           for name, path in bus_result.items()) or overwrite:
        _technology = 'BULK' if technology.upper(
        ) == 'SMARTSEQ2' else technology

        # Pipe any remote files.
        fastqs = stream_batch(
            fastqs, temp_dir=temp_dir
        ) if is_batch else stream_fastqs(
            fastqs, temp_dir=temp_dir
        )
        bus_result = kallisto_bus(
            fastqs,
            index_path,
            _technology,
            out_dir,
            threads=threads,
            paired=paired,
            genomebam=genomebam,
            aa=aa,
            strand=strand,
            gtf_path=gtf_path,
            chromosomes_path=chromosomes_path,
            inleaved=inleaved,
            demultiplexed=demultiplexed,
            batch_barcodes=batch_barcodes,
            numreads=numreads,
            n=store_num,
            lr=lr,
            lr_thresh=lr_thresh,
            lr_error_rate=lr_error_rate,
            union=union,
            no_jump=no_jump
        )
    else:
        logger.info(
            'Skipping kallisto bus because output files already exist. Use the --overwrite flag to overwrite.'
        )
    unfiltered_results.update(bus_result)

    if t2g_path.upper() == "NONE":
        tmp_t2g = os.path.join(temp_dir, "t2g.txt")
        t2g_path = make_transcript_t2g(bus_result['txnames'], tmp_t2g)

    sort_result = bustools_sort(
        bus_result['bus'],
        os.path.join(
            temp_dir,
            update_filename(os.path.basename(bus_result['bus']), SORT_CODE)
        ),
        temp_dir=temp_dir,
        threads=threads,
        memory=memory,
        store_num=store_num and not keep_flags
    )
    correct = True
    if whitelist_path and whitelist_path.upper() == "NONE":
        correct = False
    if not correct:
        whitelist_path = None
    if not whitelist_path and not demultiplexed and correct:
        logger.info('On-list not provided')
        whitelist_path = copy_or_create_whitelist(
            technology if not FB else '10xFB', sort_result['bus'], out_dir
        )
        unfiltered_results.update({'whitelist': whitelist_path})

    prev_result = sort_result
    if inspect:
        inspect_result = bustools_inspect(
            prev_result['bus'],
            os.path.join(out_dir, INSPECT_FILENAME),
            whitelist_path=whitelist_path,
        )
        unfiltered_results.update(inspect_result)
    if not demultiplexed and correct:
        prev_result = bustools_correct(
            prev_result['bus'],
            os.path.join(
                temp_dir,
                update_filename(
                    os.path.basename(prev_result['bus']), CORRECT_CODE
                )
            ), whitelist_path, False, exact_barcodes
        )
        prev_result = bustools_sort(
            prev_result['bus'],
            os.path.join(out_dir, f'output.{UNFILTERED_CODE}.bus')
            if not FB else os.path.join(
                temp_dir,
                update_filename(
                    os.path.basename(prev_result['bus']), SORT_CODE
                )
            ),
            temp_dir=temp_dir,
            threads=threads,
            memory=memory
        )
        if FB:
            logger.info(
                f'Creating {technology} feature-to-barcode map at {out_dir}'
            )
            map_path = create_10x_feature_barcode_map(
                os.path.join(out_dir, '10x_feature_barcode_map.txt')
            )
            prev_result = bustools_project(
                prev_result['bus'],
                os.path.join(
                    temp_dir,
                    update_filename(
                        os.path.basename(prev_result['bus']), PROJECT_CODE
                    )
                ), map_path, bus_result['ecmap'], bus_result['txnames']
            )
            prev_result = bustools_sort(
                prev_result['bus'],
                os.path.join(out_dir, f'output.{UNFILTERED_CODE}.bus'),
                temp_dir=temp_dir,
                threads=threads,
                memory=memory
            )

        unfiltered_results.update({'bus_scs': prev_result['bus']})

    # Helper function to update results with suffix
    def update_results_with_suffix(current_results, new_results, suffix):
        current_results.update({
            f'{key}{suffix}': value
            for key, value in new_results.items()
        })

    # Write capture file & capture internal/umi records (for SMARTSEQ3)
    capture_path = None
    if technology.upper() == 'SMARTSEQ3':
        capture_path = write_smartseq3_capture(
            os.path.join(out_dir, CAPTURE_FILENAME)
        )

    techsplit = technology.split(":")
    ignore_umis = False
    if len(techsplit) > 2 and len(
            techsplit[1]
    ) >= 2 and techsplit[1][0] == "-" and techsplit[1][1] == "1":
        ignore_umis = True
    cm = (
        technology.upper() in ('BULK', 'SMARTSEQ2', 'SMARTSEQ3')
    ) or ignore_umis
    quant = cm and tcc
    if quant_umis:
        quant = True
        no_fragment = True
    suffix_to_inspect_filename = {'': ''}
    if (technology.upper() == 'SMARTSEQ3'):
        suffix_to_inspect_filename = {
            INTERNAL_SUFFIX: INSPECT_INTERNAL_FILENAME,
            UMI_SUFFIX: INSPECT_UMI_FILENAME,
        }
    use_suffixes = len(suffix_to_inspect_filename) > 1
    replacement = replacement_path
    if use_suffixes:
        # Can't do replacements when there are suffixes (e.g. smart-seq3)
        replacement = None
    modifications = [''] if not replacement else ['', '_modified']
    for suffix, inspect_filename in suffix_to_inspect_filename.items():
        if use_suffixes:
            fname1 = os.path.join(out_dir, f'output{suffix}.bus')
            fname2 = os.path.join(
                out_dir, f'output{suffix}.{UNFILTERED_CODE}.bus'
            )
            capture_result = bustools_capture(
                prev_result['bus'],
                fname1,
                capture_path,
                capture_type='umis',
                complement=suffix == UMI_SUFFIX
            )
            update_results_with_suffix(
                unfiltered_results, capture_result, suffix
            )
            if inspect:
                inspect_result = bustools_inspect(
                    capture_result['bus'],
                    os.path.join(out_dir, inspect_filename),
                    whitelist_path=whitelist_path,
                )
                update_results_with_suffix(
                    unfiltered_results, inspect_result, suffix
                )
            sort_result = bustools_sort(
                capture_result['bus'],
                fname2,
                temp_dir=temp_dir,
                threads=threads,
                memory=memory
            )
        else:
            sort_result = prev_result
        for modified in modifications:
            if replacement and modified:
                # Replacement time, let's just replace the corrected file
                replaced_result = bustools_correct(
                    sort_result['bus'],
                    os.path.join(
                        temp_dir,
                        update_filename(
                            os.path.basename(sort_result['bus']), CORRECT_CODE
                        )
                    ), replacement, True
                )
                # Now let's create a new sort file
                sort_result = bustools_sort(
                    replaced_result['bus'],
                    os.path.join(
                        out_dir, f'output{modified}.{UNFILTERED_CODE}.bus'
                    ),
                    temp_dir=temp_dir,
                    threads=threads,
                    memory=memory
                )
                prev_result = sort_result
            counts_dir = os.path.join(
                out_dir, f'{UNFILTERED_COUNTS_DIR}{suffix}{modified}'
            )
            make_directory(counts_dir)
            quant_dir = os.path.join(
                out_dir, f'{UNFILTERED_QUANT_DIR}{suffix}{modified}'
            )
            if quant:
                make_directory(quant_dir)
            counts_prefix = os.path.join(
                counts_dir,
                TCC_PREFIX if tcc else FEATURE_PREFIX if kite else COUNTS_PREFIX
            )

            count_result = bustools_count(
                sort_result['bus'],
                counts_prefix,
                t2g_path,
                bus_result['ecmap'],
                bus_result['txnames'],
                tcc=tcc,
                mm=mm or tcc,
                cm=(suffix == INTERNAL_SUFFIX) if use_suffixes else cm,
                umi_gene=(suffix == UMI_SUFFIX) if use_suffixes else umi_gene,
                em=em,
                batch_barcodes=batch_barcodes,
            )
            update_results_with_suffix(unfiltered_results, count_result, suffix)
            quant_result = None
            if quant:
                quant_result = kallisto_quant_tcc(
                    count_result['mtx'],
                    index_path,
                    count_result['ec'],
                    t2g_path,
                    quant_dir,
                    flens_path=None if (use_suffixes and suffix == UMI_SUFFIX)
                    else bus_result.get('flens'),
                    l=fragment_l,
                    s=fragment_s,
                    threads=threads,
                    bootstraps=bootstraps,
                    matrix_to_files=matrix_to_files,
                    matrix_to_directories=matrix_to_directories,
                    no_fragment=no_fragment,
                    lr=lr,
                    lr_platform=lr_platform,
                )
                update_results_with_suffix(
                    unfiltered_results, quant_result, suffix
                )

            # Convert outputs.
            if 'genes' in count_result:
                genes_by_name_path = f'{counts_prefix}.{GENE_NAMES_FILENAME}'
                if quant:
                    genes_by_name_path = os.path.join(
                        quant_dir, ABUNDANCE_GENE_NAMES_FILENAME
                    )
                logger.info(f'Writing gene names to file {genes_by_name_path}')
                genes_by_name = obtain_gene_names(
                    t2g_path, count_result.get('genes')
                )
                if genes_by_name:
                    count_result.update({
                        'genenames':
                            write_list_to_file(
                                genes_by_name, genes_by_name_path
                            )
                    })
            update_results_with_suffix(unfiltered_results, count_result, suffix)
            final_result = quant_result if quant else count_result
            if cellranger:
                cr_result = matrix_to_cellranger(
                    count_result['mtx'], count_result['barcodes'],
                    count_result['genes'], t2g_path,
                    os.path.join(counts_dir, f'{CELLRANGER_DIR}{suffix}')
                )
                update_results_with_suffix(
                    unfiltered_results, {'cellranger': cr_result}, suffix
                )
            if loom or h5ad:
                name = GENE_NAME
                if kite:
                    name = FEATURE_NAME
                elif quant:
                    name = TRANSCRIPT_NAME
                update_results_with_suffix(
                    unfiltered_results,
                    convert_matrix(
                        quant_dir if quant else counts_dir,
                        final_result['mtx'],
                        count_result['barcodes'],
                        batch_barcodes_path=count_result['batch_barcodes']
                        if batch_barcodes else None,
                        genes_path=final_result['txnames']
                        if quant else final_result.get('genes'),
                        t2g_path=t2g_path,
                        ec_path=count_result.get('ec'),
                        txnames_path=bus_result['txnames'],
                        name=name,
                        loom=loom,
                        loom_names=loom_names,
                        h5ad=h5ad,
                        by_name=by_name,
                        tcc=tcc and not quant,
                        threads=threads,
                    ), suffix
                )

    # NOTE: bulk/smartseq2 does not support filtering, so everything here
    # assumes technology is not bulk/smartseq2
    if filter == 'bustools':
        filtered_counts_prefix = os.path.join(
            out_dir, FILTERED_COUNTS_DIR,
            TCC_PREFIX if tcc else FEATURE_PREFIX if kite else COUNTS_PREFIX
        )
        filtered_whitelist_path = os.path.join(
            out_dir, FILTER_WHITELIST_FILENAME
        )
        filtered_bus_path = os.path.join(out_dir, f'output.{FILTERED_CODE}.bus')
        if technology.upper() == 'SMARTSEQ3':
            capture_result = bustools_capture(
                prev_result['bus'],
                os.path.join(out_dir, f'output.{FILTERED_CODE}.umi.bus'),
                capture_path,
                capture_type='umis',
                complement=True
            )
            prev_result = capture_result
        results['filtered'] = filter_with_bustools(
            prev_result['bus'],
            bus_result['ecmap'],
            bus_result['txnames'],
            t2g_path,
            filtered_whitelist_path,
            filtered_bus_path,
            filter_threshold=filter_threshold,
            counts_prefix=filtered_counts_prefix,
            kite=kite,
            tcc=tcc,
            temp_dir=temp_dir,
            threads=threads,
            memory=memory,
            loom=loom,
            loom_names=loom_names,
            h5ad=h5ad,
            by_name=by_name,
            umi_gene=umi_gene,
            em=em,
        )

    # Generate report.
    STATS.end()
    stats_path = STATS.save(os.path.join(out_dir, KB_INFO_FILENAME))
    results.update({'stats': stats_path})
    if report:
        nb_path = os.path.join(out_dir, REPORT_NOTEBOOK_FILENAME)
        html_path = os.path.join(out_dir, REPORT_HTML_FILENAME)
        logger.info(
            f'Writing report Jupyter notebook at {nb_path} and rendering it to {html_path}'
        )
        suffix = ""
        if technology.upper() == 'SMARTSEQ3':
            suffix = UMI_SUFFIX
        report_result = render_report(
            stats_path,
            bus_result['info'],
            unfiltered_results[f'inspect{suffix}'],
            nb_path,
            html_path,
            unfiltered_results[f'mtx{suffix}'],
            unfiltered_results.get(f'barcodes{suffix}'),
            unfiltered_results.get(f'genes{suffix}'),
            t2g_path,
            temp_dir=temp_dir
        )
        unfiltered_results.update(report_result)

    return results


@logger.namespaced('count_nac')
def count_nac(
    index_path: str,
    t2g_path: str,
    cdna_t2c_path: str,
    intron_t2c_path: str,
    technology: str,
    out_dir: str,
    fastqs: List[str],
    whitelist_path: Optional[str] = None,
    replacement_path: Optional[str] = None,
    tcc: bool = False,
    mm: bool = False,
    filter: Optional[Literal['bustools']] = None,
    filter_threshold: Optional[int] = None,
    temp_dir: str = 'tmp',
    threads: int = 8,
    memory: str = '4G',
    overwrite: bool = False,
    loom: bool = False,
    loom_names: List[str] = ['barcode', 'target_name'],
    h5ad: bool = False,
    by_name: bool = False,
    cellranger: bool = False,
    inspect: bool = True,
    report: bool = False,
    nucleus: bool = False,
    fragment_l: Optional[int] = None,
    fragment_s: Optional[int] = None,
    paired: bool = False,
    genomebam: bool = False,
    strand: Optional[Literal['unstranded', 'forward', 'reverse']] = None,
    umi_gene: bool = True,
    em: bool = False,
    sum_matrices: Optional[Literal['none', 'cell', 'nucleus', 'total']] = None,
    gtf_path: Optional[str] = None,
    chromosomes_path: Optional[str] = None,
    inleaved: bool = False,
    demultiplexed: bool = False,
    batch_barcodes: bool = False,
    numreads: int = None,
    store_num: bool = False,
    lr: bool = False,
    lr_thresh: float = 0.8,
    lr_error_rate: float = None,
    lr_platform: str = 'ONT',
    union: bool = False,
    no_jump: bool = False,
    quant_umis: bool = False,
    keep_flags: bool = False,
    exact_barcodes: bool = False,
) -> Dict[str, Union[Dict[str, str], str]]:
    """Generates RNA velocity matrices for single-cell RNA seq.

    Args:
        index_path: Path to kallisto index
        t2g_path: Path to transcript-to-gene mapping
        cdna_t2c_path: Path to cDNA transcripts-to-capture file
        intron_t2c_path: Path to intron transcripts-to-capture file
        technology: Single-cell technology used
        out_dir: Path to output directory
        fastqs: List of FASTQ file paths or a single batch definition file
        whitelist_path: Path to whitelist, defaults to `None`
        replacement_path: Path to replacement list, defaults to `None`
        tcc: Whether to generate a TCC matrix instead of a gene count matrix,
            defaults to `False`
        mm: Whether to include BUS records that pseudoalign to multiple genes,
            defaults to `False`
        filter: Filter to use to generate a filtered count matrix,
            defaults to `None`
        filter_threshold: Barcode filter threshold for bustools, defaults
            to `None`
        temp_dir: Path to temporary directory, defaults to `tmp`
        threads: Number of threads to use, defaults to `8`
        memory: Amount of memory to use, defaults to `4G`
        overwrite: Overwrite an existing index file, defaults to `False`
        loom: Whether to convert the final count matrix into a loom file,
            defaults to `False`
        loom_names: Names for col_attrs and row_attrs in loom file,
            defaults to `['barcode','target_name']`
        h5ad: Whether to convert the final count matrix into a h5ad file,
            defaults to `False`
        by_name: Aggregate counts by name instead of ID.
        cellranger: Whether to convert the final count matrix into a
            cellranger-compatible matrix, defaults to `False`
        inspect: Whether or not to inspect the output BUS file and generate
            the inspect.json
        report: Generate HTML reports, defaults to `False`
        nucleus: Whether this is a single-nucleus experiment. if `True`, the
            spliced and unspliced count matrices will be summed, defaults to
            `False`
        fragment_l: Mean length of fragments, defaults to `None`
        fragment_s: Standard deviation of fragment lengths, defaults to `None`
        paired: Whether the fastqs are paired. Has no effect when a single
            batch file is provided. Defaults to `False`
        genomebam: Project pseudoalignments to genome sorted BAM file, defaults to
            `False`
        strand: Strandedness, defaults to `None`
        umi_gene: Whether to perform gene-level UMI collapsing, defaults to
            `True`
        em: Whether to estimate gene abundances using EM algorithm, defaults to
            `False`
        sum_matrices: How to sum output matrices, defaults to `None`
        gtf_path: GTF file for transcriptome information (required for --genomebam),
            defaults to `None`
        chromosomes_path: Tab separated file with chromosome names and lengths
            (optional for --genomebam, but recommended), defaults to `None`
        inleaved: Whether input FASTQ is interleaved, defaults to `False`
        demultiplexed: Whether FASTQs are demultiplexed, defaults to `False`
        batch_barcodes: Whether sample ID should be in barcode, defaults to `False`
        numreads: Maximum number of reads to process from supplied input
        store_num: Whether to store read numbers in BUS file, defaults to `False`
        lr: Whether to use lr-kallisto in read mapping, defaults to `False`
        lr_thresh: Sets the --threshold for lr-kallisto, defaults to `0.8`
        lr_error_rate: Sets the --error-rate for lr-kallisto, defaults to `None`
        lr_platform: Sets the --platform for lr-kallisto, defaults to `ONT`
        union: Use set union for pseudoalignment, defaults to `False`
        no_jump: Disable pseudoalignment "jumping", defaults to `False`
        quant_umis: Whether to run quant-tcc when there are UMIs, defaults to `False`
        keep_flags: Preserve flag column when sorting BUS file, defaults to `False`
        exact_barcodes: Use exact match for 'correcting' barcodes to on-list, defaults to `False`

    Returns:
        Dictionary containing path to generated index
    """
    STATS.start()
    is_batch = isinstance(fastqs, str)

    results = {}
    make_directory(out_dir)
    unfiltered_results = results.setdefault('unfiltered', {})

    bus_result = {
        'bus': os.path.join(out_dir, BUS_FILENAME),
        'ecmap': os.path.join(out_dir, ECMAP_FILENAME),
        'txnames': os.path.join(out_dir, TXNAMES_FILENAME),
        'info': os.path.join(out_dir, KALLISTO_INFO_FILENAME)
    }
    if technology.upper() in ('BULK', 'SMARTSEQ2', 'SMARTSEQ3'):
        bus_result['saved_index'] = os.path.join(out_dir, SAVED_INDEX_FILENAME)
        if technology.upper() == 'SMARTSEQ3':
            bus_result['flens'] = os.path.join(out_dir, FLENS_FILENAME)
            paired = True
    if any(not os.path.exists(path)
           for name, path in bus_result.items()) or overwrite:
        _technology = 'BULK' if technology.upper(
        ) == 'SMARTSEQ2' else technology
        # Pipe any remote files.
        fastqs = stream_batch(
            fastqs, temp_dir=temp_dir
        ) if is_batch else stream_fastqs(
            fastqs, temp_dir=temp_dir
        )
        bus_result = kallisto_bus(
            fastqs,
            index_path,
            _technology,
            out_dir,
            threads=threads,
            paired=paired,
            genomebam=genomebam,
            strand=strand,
            gtf_path=gtf_path,
            chromosomes_path=chromosomes_path,
            inleaved=inleaved,
            demultiplexed=demultiplexed,
            batch_barcodes=batch_barcodes,
            numreads=numreads,
            n=store_num,
            lr=lr,
            lr_thresh=lr_thresh,
            lr_error_rate=lr_error_rate,
            union=union,
            no_jump=no_jump
        )
    else:
        logger.info(
            'Skipping kallisto bus because output files already exist. Use the --overwrite flag to overwrite.'
        )
    unfiltered_results.update(bus_result)

    if t2g_path.upper() == "NONE":
        tmp_t2g = os.path.join(temp_dir, "t2g.txt")
        t2g_path = make_transcript_t2g(bus_result['txnames'], tmp_t2g)

    sort_result = bustools_sort(
        bus_result['bus'],
        os.path.join(
            temp_dir,
            update_filename(os.path.basename(bus_result['bus']), SORT_CODE)
        ),
        temp_dir=temp_dir,
        threads=threads,
        memory=memory,
        store_num=store_num and not keep_flags
    )
    correct = True
    if whitelist_path and whitelist_path.upper() == "NONE":
        correct = False
    if not correct:
        whitelist_path = None
    if not whitelist_path and not demultiplexed and correct:
        logger.info('On-list not provided')
        whitelist_path = copy_or_create_whitelist(
            technology, sort_result['bus'], out_dir
        )
        unfiltered_results.update({'whitelist': whitelist_path})

    if inspect:
        inspect_result = bustools_inspect(
            sort_result['bus'],
            os.path.join(out_dir, INSPECT_FILENAME),
            whitelist_path=whitelist_path,
        )
        unfiltered_results.update(inspect_result)

    prev_result = sort_result
    if not demultiplexed and correct:
        prev_result = bustools_correct(
            prev_result['bus'],
            os.path.join(
                temp_dir,
                update_filename(
                    os.path.basename(sort_result['bus']), CORRECT_CODE
                )
            ), whitelist_path, False, exact_barcodes
        )
        prev_result = bustools_sort(
            prev_result['bus'],
            os.path.join(out_dir, f'output.{UNFILTERED_CODE}.bus'),
            temp_dir=temp_dir,
            threads=threads,
            memory=memory
        )
        unfiltered_results.update({'bus_scs': prev_result['bus']})

    # Helper function to update results with suffix
    def update_results_with_suffix(current_results, new_results, suffix):
        current_results.update({
            f'{key}{suffix}': value
            for key, value in new_results.items()
        })

    # Write capture file & capture internal/umi records (for SMARTSEQ3)
    capture_path = None
    if technology.upper() == 'SMARTSEQ3':
        capture_path = write_smartseq3_capture(
            os.path.join(out_dir, CAPTURE_FILENAME)
        )

    techsplit = technology.split(":")
    ignore_umis = False
    if len(techsplit) > 2 and len(
            techsplit[1]
    ) >= 2 and techsplit[1][0] == "-" and techsplit[1][1] == "1":
        ignore_umis = True
    cm = (
        technology.upper() in ('BULK', 'SMARTSEQ2', 'SMARTSEQ3')
    ) or ignore_umis
    quant = cm and tcc
    suffix_to_inspect_filename = {'': ''}
    if (technology.upper() == 'SMARTSEQ3'):
        suffix_to_inspect_filename = {
            INTERNAL_SUFFIX: INSPECT_INTERNAL_FILENAME,
            UMI_SUFFIX: INSPECT_UMI_FILENAME,
        }
    use_suffixes = len(suffix_to_inspect_filename) > 1
    replacement = replacement_path
    if use_suffixes:
        # Can't do replacements when there are suffixes (e.g. smart-seq3)
        replacement = None
    modifications = [''] if not replacement else ['', '_modified']
    for suffix, inspect_filename in suffix_to_inspect_filename.items():
        if use_suffixes:
            capture_result = bustools_capture(
                prev_result['bus'],
                os.path.join(out_dir, f'output{suffix}.bus'),
                capture_path,
                capture_type='umis',
                complement=suffix == UMI_SUFFIX
            )
            update_results_with_suffix(
                unfiltered_results, capture_result, suffix
            )
            if inspect:
                inspect_result = bustools_inspect(
                    capture_result['bus'],
                    os.path.join(out_dir, inspect_filename),
                    whitelist_path=whitelist_path,
                )
                update_results_with_suffix(
                    unfiltered_results, inspect_result, suffix
                )
            sort_result = bustools_sort(
                capture_result['bus'],
                os.path.join(out_dir, f'output{suffix}.{UNFILTERED_CODE}.bus'),
                temp_dir=temp_dir,
                threads=threads,
                memory=memory
            )
        else:
            sort_result = prev_result
        for modified in modifications:
            if replacement and modified:
                # Replacement time, let's just replace the corrected file
                replaced_result = bustools_correct(
                    sort_result['bus'],
                    os.path.join(
                        temp_dir,
                        update_filename(
                            os.path.basename(sort_result['bus']), CORRECT_CODE
                        )
                    ), replacement, True
                )
                # Now let's create a new sort file
                sort_result = bustools_sort(
                    replaced_result['bus'],
                    os.path.join(
                        out_dir, f'output{modified}.{UNFILTERED_CODE}.bus'
                    ),
                    temp_dir=temp_dir,
                    threads=threads,
                    memory=memory
                )
                prev_result = sort_result
            counts_dir = os.path.join(
                out_dir, f'{UNFILTERED_COUNTS_DIR}{suffix}{modified}'
            )
            make_directory(counts_dir)
            quant_dir = os.path.join(
                out_dir, f'{UNFILTERED_QUANT_DIR}{suffix}{modified}'
            )
            if quant:
                make_directory(quant_dir)
            counts_prefix = os.path.join(
                counts_dir, TCC_PREFIX if tcc else COUNTS_PREFIX
            )
            count_result = bustools_count(
                sort_result['bus'],
                counts_prefix,
                t2g_path,
                bus_result['ecmap'],
                bus_result['txnames'],
                tcc=tcc,
                mm=mm or tcc,
                cm=(suffix == INTERNAL_SUFFIX) if use_suffixes else cm,
                umi_gene=(suffix == UMI_SUFFIX) if use_suffixes else umi_gene,
                em=em,
                nascent_path=intron_t2c_path,
                batch_barcodes=batch_barcodes,
            )
            count_result = count_result_to_dict(count_result)
            prefixes = ['processed', 'unprocessed', 'ambiguous']  # 0,1,2
            for i in range(len(prefixes)):
                prefix = prefixes[i]
                if i == 0 and 'genes' in count_result[i]:
                    # Only need to write this once
                    genes_by_name_path = f'{counts_prefix}.{GENE_NAMES_FILENAME}'
                    logger.info(
                        f'Writing gene names to file {genes_by_name_path}'
                    )
                    genes_by_name = obtain_gene_names(
                        t2g_path, count_result[i].get('genes')
                    )
                    if genes_by_name:
                        count_result[i].update({
                            'genenames':
                                write_list_to_file(
                                    genes_by_name, genes_by_name_path
                                )
                        })
                prefix_results = unfiltered_results.setdefault(prefix, {})
                update_results_with_suffix(prefix_results, sort_result, suffix)
                update_results_with_suffix(
                    prefix_results, count_result[i], suffix
                )
                if cellranger:
                    cr_result = matrix_to_cellranger(
                        count_result[i]['mtx'], count_result[i]['barcodes'],
                        count_result[i]['genes'], t2g_path,
                        os.path.join(
                            counts_dir, f'{CELLRANGER_DIR}_{prefix}{suffix}'
                        )
                    )
                    update_results_with_suffix(
                        prefix_results, {'cellranger': cr_result}, suffix
                    )
            if sum_matrices and sum_matrices != 'none':
                # Sum up multiple matrices
                sums = {}
                if sum_matrices == 'cell' or sum_matrices == 'total':
                    sums['cell'] = do_sum_matrices(
                        count_result[prefixes.index('processed')]['mtx'],
                        count_result[prefixes.index('ambiguous')]['mtx'],
                        f'{counts_prefix}.cell.mtx', em or mm
                    )
                if sum_matrices == 'nucleus' or sum_matrices == 'total':
                    sums['nucleus'] = do_sum_matrices(
                        count_result[prefixes.index('unprocessed')]['mtx'],
                        count_result[prefixes.index('ambiguous')]['mtx'],
                        f'{counts_prefix}.nucleus.mtx', em or mm
                    )
                if sum_matrices == 'total':
                    sums['total'] = do_sum_matrices(
                        f'{counts_prefix}.mature.mtx',
                        f'{counts_prefix}.nucleus.mtx',
                        f'{counts_prefix}.total.mtx', em or mm
                    )
                for prefix, f in sums.items():
                    res = copy.deepcopy(count_result[0])
                    res['mtx'] = f
                    prefix_results = unfiltered_results.setdefault(prefix, {})
                    update_results_with_suffix(
                        prefix_results, sort_result, suffix
                    )
                    update_results_with_suffix(prefix_results, res, suffix)
                    if cellranger:
                        cr_result = matrix_to_cellranger(
                            res['mtx'], res['barcodes'], res['genes'], t2g_path,
                            os.path.join(
                                counts_dir, f'{CELLRANGER_DIR}_{prefix}{suffix}'
                            )
                        )
                        update_results_with_suffix(
                            prefix_results, {'cellranger': cr_result}, suffix
                        )

            if loom or h5ad:
                name = GENE_NAME
                if quant:
                    name = TRANSCRIPT_NAME

                convert_result = convert_matrices(
                    quant_dir if quant else counts_dir,
                    [
                        unfiltered_results[prefix][f'mtx{suffix}']
                        for prefix in prefixes
                    ],
                    [
                        unfiltered_results[prefix][f'barcodes{suffix}']
                        for prefix in prefixes
                    ],
                    [
                        unfiltered_results[prefix][f'batch_barcodes{suffix}']
                        if batch_barcodes else None for prefix in prefixes
                    ],
                    genes_paths=[
                        unfiltered_results[prefix][f'ec{suffix}'] if tcc else
                        unfiltered_results[prefix].get(f'genes{suffix}')
                        for prefix in prefixes
                    ],
                    t2g_path=t2g_path,
                    ec_paths=[
                        unfiltered_results[prefix].get(f'ec{suffix}')
                        for prefix in prefixes
                    ],
                    txnames_path=bus_result['txnames'],
                    name=name,
                    loom=loom,
                    loom_names=loom_names,
                    h5ad=h5ad,
                    by_name=by_name,
                    tcc=False,
                    threads=threads,
                )
                update_results_with_suffix(
                    unfiltered_results, convert_result, suffix
                )

    # NOTE: bulk/smartseq2 does not support filtering, so everything here
    # assumes technology is not bulk/smartseq2
    if filter:
        filtered_results = results.setdefault('filtered', {})
        if filter == 'bustools':
            if technology.upper() == 'SMARTSEQ3':
                capture_result = bustools_capture(
                    prev_result['bus'],
                    os.path.join(out_dir, f'output.{FILTERED_CODE}.umi.bus'),
                    capture_path,
                    capture_type='umis',
                    complement=True
                )
                prev_result = capture_result
            filtered_results.update(
                filter_with_bustools(
                    prev_result['bus'],
                    bus_result['ecmap'],
                    bus_result['txnames'],
                    t2g_path,
                    os.path.join(out_dir, FILTER_WHITELIST_FILENAME),
                    os.path.join(out_dir, f'output.{FILTERED_CODE}.bus'),
                    filter_threshold=filter_threshold,
                    temp_dir=temp_dir,
                    memory=memory,
                    count=False,
                    umi_gene=umi_gene,
                    em=em,
                )
            )

            filtered_counts_dir = os.path.join(out_dir, FILTERED_COUNTS_DIR)
            make_directory(filtered_counts_dir)
            filtered_counts_prefix = os.path.join(
                filtered_counts_dir, TCC_PREFIX if tcc else COUNTS_PREFIX
            )
            count_result = bustools_count(
                filtered_results['bus_scs'],
                filtered_counts_prefix,
                t2g_path,
                bus_result['ecmap'],
                bus_result['txnames'],
                tcc=tcc,
                mm=mm or tcc,
                cm=False,
                umi_gene=umi_gene,
                em=em,
                nascent_path=intron_t2c_path,
            )
            count_result = count_result_to_dict(count_result)
            prefixes = ['processed', 'unprocessed', 'ambiguous']
            for i in range(len(prefixes)):
                prefix = prefixes[i]
                filtered_results[prefix] = {}
                if i == 0 and 'genes' in filtered_results[prefix]:
                    # Only need to write this once
                    genes_by_name_path = f'{filtered_counts_prefix}.{GENE_NAMES_FILENAME}'
                    logger.info(
                        f'Writing gene names to file {genes_by_name_path}'
                    )
                    genes_by_name = obtain_gene_names(
                        t2g_path, filtered_results[prefix].get('genes')
                    )
                    if genes_by_name:
                        filtered_results[prefix].update({
                            'genenames':
                                write_list_to_file(
                                    genes_by_name, genes_by_name_path
                                )
                        })
                if cellranger:
                    cr_result = matrix_to_cellranger(
                        count_result[i]['mtx'], count_result[i]['barcodes'],
                        count_result[i]['genes'], t2g_path,
                        os.path.join(
                            filtered_counts_dir, f'{CELLRANGER_DIR}_{prefix}'
                        )
                    )
                    filtered_results[prefix].update({'cellranger': cr_result})
                filtered_results[prefix].update(count_result[i])

            if sum_matrices and sum_matrices != 'none':
                # Sum up multiple matrices
                sums = {}
                if sum_matrices == 'cell' or sum_matrices == 'total':
                    sums['cell'] = do_sum_matrices(
                        count_result[prefixes.index('processed')]['mtx'],
                        count_result[prefixes.index('ambiguous')]['mtx'],
                        f'{filtered_counts_prefix}.cell.mtx', em or mm
                    )
                if sum_matrices == 'nucleus' or sum_matrices == 'total':
                    sums['nucleus'] = do_sum_matrices(
                        count_result[prefixes.index('unprocessed')]['mtx'],
                        count_result[prefixes.index('ambiguous')]['mtx'],
                        f'{filtered_counts_prefix}.nucleus.mtx', em or mm
                    )
                if sum_matrices == 'total':
                    sums['total'] = do_sum_matrices(
                        f'{filtered_counts_prefix}.mature.mtx',
                        f'{filtered_counts_prefix}.nucleus.mtx',
                        f'{filtered_counts_prefix}.total.mtx', em or mm
                    )
                for prefix, f in sums.items():
                    res = copy.deepcopy(count_result[0])
                    res['mtx'] = f
                    filtered_results[prefix] = {}
                    if cellranger:
                        cr_result = matrix_to_cellranger(
                            res['mtx'], res['barcodes'], res['genes'], t2g_path,
                            os.path.join(
                                filtered_counts_dir,
                                f'{CELLRANGER_DIR}_{prefix}'
                            )
                        )
                        filtered_results[prefix].update({
                            'cellranger': cr_result
                        })
                    filtered_results[prefix].update(res)

        if loom or h5ad:
            filtered_results.update(
                convert_matrices(
                    filtered_counts_dir,
                    [filtered_results[prefix]['mtx'] for prefix in prefixes],
                    [
                        filtered_results[prefix]['barcodes']
                        for prefix in prefixes
                    ],
                    [
                        filtered_results[prefix]['batch_barcodes']
                        if batch_barcodes else None for prefix in prefixes
                    ],
                    genes_paths=[
                        filtered_results[prefix].get('genes')
                        for prefix in prefixes
                    ],
                    t2g_path=t2g_path,
                    ec_paths=[
                        filtered_results[prefix].get('ec')
                        for prefix in prefixes
                    ],
                    txnames_path=bus_result['txnames'],
                    loom=loom,
                    loom_names=loom_names,
                    h5ad=h5ad,
                    by_name=by_name,
                    tcc=tcc,
                    nucleus=nucleus,
                    threads=threads,
                )
            )

    STATS.end()
    stats_path = STATS.save(os.path.join(out_dir, KB_INFO_FILENAME))
    results.update({'stats': stats_path})

    # Reports
    nb_path = os.path.join(out_dir, REPORT_NOTEBOOK_FILENAME)
    html_path = os.path.join(out_dir, REPORT_HTML_FILENAME)
    if report:
        logger.info(
            f'Writing report Jupyter notebook at {nb_path} and rendering it to {html_path}'
        )

        for prefix in prefixes:
            nb_path = os.path.join(
                out_dir, update_filename(REPORT_NOTEBOOK_FILENAME, prefix)
            )
            html_path = os.path.join(
                out_dir, update_filename(REPORT_HTML_FILENAME, prefix)
            )
            logger.info(
                f'Writing report Jupyter notebook at {nb_path} and rendering it to {html_path}'
            )
            suffix = ""
            if technology.upper() == 'SMARTSEQ3':
                suffix = UMI_SUFFIX
            report_result = render_report(
                stats_path,
                bus_result['info'],
                unfiltered_results[prefix][f'inspect{suffix}'],
                nb_path,
                html_path,
                unfiltered_results[prefix][f'mtx{suffix}'],
                unfiltered_results[prefix].get(f'barcodes{suffix}'),
                unfiltered_results[prefix].get(f'genes{suffix}'),
                t2g_path,
                temp_dir=temp_dir
            )
            unfiltered_results[prefix].update(report_result)
        if tcc:
            logger.warning(
                'Plots for TCC matrices have not yet been implemented. The HTML report will not contain any plots.'
            )

    return results


@logger.namespaced('count_lamanno')
def count_velocity(
    index_path: str,
    t2g_path: str,
    cdna_t2c_path: str,
    intron_t2c_path: str,
    technology: str,
    out_dir: str,
    fastqs: List[str],
    whitelist_path: Optional[str] = None,
    tcc: bool = False,
    mm: bool = False,
    filter: Optional[Literal['bustools']] = None,
    filter_threshold: Optional[int] = None,
    temp_dir: str = 'tmp',
    threads: int = 8,
    memory: str = '4G',
    overwrite: bool = False,
    loom: bool = False,
    h5ad: bool = False,
    by_name: bool = False,
    cellranger: bool = False,
    inspect: bool = True,
    report: bool = False,
    nucleus: bool = False,
    fragment_l: Optional[int] = None,
    fragment_s: Optional[int] = None,
    paired: bool = False,
    strand: Optional[Literal['unstranded', 'forward', 'reverse']] = None,
    umi_gene: bool = False,
    em: bool = False,
) -> Dict[str, Union[Dict[str, str], str]]:
    """Generates RNA velocity matrices (DEPRECATED).

    Args:
        index_path: Path to kallisto index
        t2g_path: Path to transcript-to-gene mapping
        cdna_t2c_path: Path to cDNA transcripts-to-capture file
        intron_t2c_path: Path to intron transcripts-to-capture file
        technology: Single-cell technology used
        out_dir: Path to output directory
        fastqs: List of FASTQ file paths or a single batch definition file
        whitelist_path: Path to whitelist, defaults to `None`
        tcc: Whether to generate a TCC matrix instead of a gene count matrix,
            defaults to `False`
        mm: Whether to include BUS records that pseudoalign to multiple genes,
            defaults to `False`
        filter: Filter to use to generate a filtered count matrix,
            defaults to `None`
        filter_threshold: Barcode filter threshold for bustools, defaults
            to `None`
        temp_dir: Path to temporary directory, defaults to `tmp`
        threads: Number of threads to use, defaults to `8`
        memory: Amount of memory to use, defaults to `4G`
        overwrite: Overwrite an existing index file, defaults to `False`
        loom: Whether to convert the final count matrix into a loom file,
            defaults to `False`
        h5ad: Whether to convert the final count matrix into a h5ad file,
            defaults to `False`
        by_name: Aggregate counts by name instead of ID. Only affects when
            `tcc=False`.
        cellranger: Whether to convert the final count matrix into a
            cellranger-compatible matrix, defaults to `False`
        inspect: Whether or not to inspect the output BUS file and generate
            the inspect.json
        report: Generate HTML reports, defaults to `False`
        nucleus: Whether this is a single-nucleus experiment. if `True`, the
            spliced and unspliced count matrices will be summed, defaults to
            `False`
        fragment_l: Mean length of fragments, defaults to `None`
        fragment_s: Standard deviation of fragment lengths, defaults to `None`
        paired: Whether the fastqs are paired. Has no effect when a single
            batch file is provided. Defaults to `False`
        strand: Strandedness, defaults to `None`
        umi_gene: Whether to perform gene-level UMI collapsing, defaults to
            `False`
        em: Whether to estimate gene abundances using EM algorithm, defaults to
            `False`

    Returns:
        Dictionary containing path to generated index
    """
    STATS.start()
    is_batch = isinstance(fastqs, str)
    BUS_CDNA_PREFIX = 'spliced'
    BUS_INTRON_PREFIX = 'unspliced'

    results = {}
    make_directory(out_dir)
    unfiltered_results = results.setdefault('unfiltered', {})

    bus_result = {
        'bus': os.path.join(out_dir, BUS_FILENAME),
        'ecmap': os.path.join(out_dir, ECMAP_FILENAME),
        'txnames': os.path.join(out_dir, TXNAMES_FILENAME),
        'info': os.path.join(out_dir, KALLISTO_INFO_FILENAME)
    }
    if technology.upper() in ('BULK', 'SMARTSEQ2'):
        bus_result['saved_index'] = os.path.join(out_dir, SAVED_INDEX_FILENAME)
    if any(not os.path.exists(path)
           for name, path in bus_result.items()) or overwrite:
        _technology = 'BULK' if technology.upper(
        ) == 'SMARTSEQ2' else technology
        # Pipe any remote files.
        fastqs = stream_batch(
            fastqs, temp_dir=temp_dir
        ) if is_batch else stream_fastqs(
            fastqs, temp_dir=temp_dir
        )
        bus_result = kallisto_bus(
            fastqs,
            index_path,
            _technology,
            out_dir,
            threads=threads,
            paired=paired,
            strand=strand
        )
    else:
        logger.info(
            'Skipping kallisto bus because output files already exist. Use the --overwrite flag to overwrite.'
        )
    unfiltered_results.update(bus_result)

    sort_result = bustools_sort(
        bus_result['bus'],
        os.path.join(
            temp_dir,
            update_filename(os.path.basename(bus_result['bus']), SORT_CODE)
        ),
        temp_dir=temp_dir,
        threads=threads,
        memory=memory
    )
    if not whitelist_path and not is_batch:
        logger.info('On-list not provided')
        whitelist_path = copy_or_create_whitelist(
            technology, sort_result['bus'], out_dir
        )
        unfiltered_results.update({'whitelist': whitelist_path})

    if inspect:
        inspect_result = bustools_inspect(
            sort_result['bus'],
            os.path.join(out_dir, INSPECT_FILENAME),
            whitelist_path=whitelist_path,
        )
        unfiltered_results.update(inspect_result)

    prev_result = sort_result
    if not is_batch:
        prev_result = bustools_correct(
            prev_result['bus'],
            os.path.join(
                temp_dir,
                update_filename(
                    os.path.basename(sort_result['bus']), CORRECT_CODE
                )
            ), whitelist_path
        )
        prev_result = bustools_sort(
            prev_result['bus'],
            os.path.join(out_dir, f'output.{UNFILTERED_CODE}.bus'),
            temp_dir=temp_dir,
            threads=threads,
            memory=memory
        )
        unfiltered_results.update({'bus_scs': prev_result['bus']})

    prefixes = [BUS_CDNA_PREFIX, BUS_INTRON_PREFIX]
    # The prefix and t2cs are swapped because we call bustools capture with
    # the --complement flag.
    prefix_to_t2c = {
        BUS_CDNA_PREFIX: intron_t2c_path,
        BUS_INTRON_PREFIX: cdna_t2c_path,
    }
    counts_dir = os.path.join(out_dir, UNFILTERED_COUNTS_DIR)
    make_directory(counts_dir)
    cm = technology.upper() in ('BULK', 'SMARTSEQ2')
    quant = cm and tcc
    if quant:
        quant_dir = os.path.join(out_dir, UNFILTERED_QUANT_DIR)
        make_directory(quant_dir)
    for prefix, t2c_path in prefix_to_t2c.items():
        capture_result = bustools_capture(
            prev_result['bus'], os.path.join(temp_dir, '{}.bus'.format(prefix)),
            t2c_path, bus_result['ecmap'], bus_result['txnames']
        )
        sort_result = bustools_sort(
            capture_result['bus'],
            os.path.join(out_dir, f'{prefix}.{UNFILTERED_CODE}.bus'),
            temp_dir=temp_dir,
            threads=threads,
            memory=memory
        )

        if prefix not in unfiltered_results:
            unfiltered_results[prefix] = {}
        unfiltered_results[prefix].update(sort_result)

        if inspect:
            inspect_result = bustools_inspect(
                sort_result['bus'],
                os.path.join(
                    out_dir, update_filename(INSPECT_FILENAME, prefix)
                ),
                whitelist_path=whitelist_path,
            )
            unfiltered_results[prefix].update(inspect_result)

        counts_prefix = os.path.join(counts_dir, prefix)
        count_result = bustools_count(
            sort_result['bus'],
            counts_prefix,
            t2g_path,
            bus_result['ecmap'],
            bus_result['txnames'],
            tcc=tcc,
            mm=mm or tcc,
            cm=cm,
            umi_gene=umi_gene,
            em=em,
        )
        unfiltered_results[prefix].update(count_result)
        if quant:
            quant_result = kallisto_quant_tcc(
                count_result['mtx'],
                index_path,
                bus_result['ecmap'],
                t2g_path,
                quant_dir,
                flens_path=bus_result.get('flens'),
                l=fragment_l,
                s=fragment_s,
                threads=threads,
            )
            unfiltered_results.update(quant_result)

        if cellranger:
            cr_result = matrix_to_cellranger(
                count_result['mtx'], count_result['barcodes'],
                count_result['genes'], t2g_path,
                os.path.join(counts_dir, f'{CELLRANGER_DIR}_{prefix}')
            )
            unfiltered_results[prefix].update({'cellranger': cr_result})

    if loom or h5ad:
        name = GENE_NAME
        if quant:
            name = TRANSCRIPT_NAME

        unfiltered_results.update(
            convert_matrices(
                quant_dir if quant else counts_dir,
                [unfiltered_results[prefix]['mtx'] for prefix in prefixes],
                [unfiltered_results[prefix]['barcodes'] for prefix in prefixes],
                genes_paths=[
                    unfiltered_results[prefix]['txnames']
                    if quant else unfiltered_results[prefix].get('genes')
                    for prefix in prefixes
                ],
                t2g_path=t2g_path,
                ec_paths=[
                    unfiltered_results[prefix].get('ec') for prefix in prefixes
                ],
                txnames_path=bus_result['txnames'],
                name=name,
                loom=loom,
                h5ad=h5ad,
                by_name=by_name,
                tcc=tcc,
                nucleus=nucleus,
                threads=threads,
            )
        )

    # NOTE: bulk/smartseq2 does not support filtering, so everything here
    # assumes technology is not bulk/smartseq2
    if filter:
        filtered_results = results.setdefault('filtered', {})
        if filter == 'bustools':
            filtered_results.update(
                filter_with_bustools(
                    prev_result['bus'],
                    bus_result['ecmap'],
                    bus_result['txnames'],
                    t2g_path,
                    os.path.join(out_dir, FILTER_WHITELIST_FILENAME),
                    os.path.join(out_dir, f'output.{FILTERED_CODE}.bus'),
                    filter_threshold=filter_threshold,
                    temp_dir=temp_dir,
                    memory=memory,
                    count=False,
                    umi_gene=umi_gene,
                    em=em,
                )
            )

            for prefix, t2c_path in prefix_to_t2c.items():
                filtered_capture_result = bustools_capture(
                    filtered_results['bus_scs'],
                    os.path.join(temp_dir, '{}.bus'.format(prefix)), t2c_path,
                    bus_result['ecmap'], bus_result['txnames']
                )
                filtered_sort_result = bustools_sort(
                    filtered_capture_result['bus'],
                    os.path.join(out_dir, f'{prefix}.{FILTERED_CODE}.bus'),
                    temp_dir=temp_dir,
                    threads=threads,
                    memory=memory
                )

                filtered_results.setdefault(prefix,
                                            {}).update(filtered_sort_result)

                filtered_counts_dir = os.path.join(out_dir, FILTERED_COUNTS_DIR)
                make_directory(filtered_counts_dir)
                filtered_counts_prefix = os.path.join(
                    filtered_counts_dir, prefix
                )
                count_result = bustools_count(
                    filtered_sort_result['bus'],
                    filtered_counts_prefix,
                    t2g_path,
                    bus_result['ecmap'],
                    bus_result['txnames'],
                    tcc=tcc,
                    mm=mm or tcc,
                    umi_gene=umi_gene,
                    em=em,
                )
                filtered_results[prefix].update(count_result)

                if cellranger:
                    if not tcc:
                        cr_result = matrix_to_cellranger(
                            count_result['mtx'], count_result['barcodes'],
                            count_result['genes'], t2g_path,
                            os.path.join(
                                filtered_counts_dir,
                                f'{CELLRANGER_DIR}_{prefix}'
                            )
                        )
                        unfiltered_results[prefix].update({
                            'cellranger': cr_result
                        })
                    else:
                        logger.warning(
                            'TCC matrices can not be converted to cellranger-compatible format.'
                        )

        if loom or h5ad:
            filtered_results.update(
                convert_matrices(
                    filtered_counts_dir,
                    [filtered_results[prefix]['mtx'] for prefix in prefixes],
                    [
                        filtered_results[prefix]['barcodes']
                        for prefix in prefixes
                    ],
                    genes_paths=[
                        filtered_results[prefix].get('genes')
                        for prefix in prefixes
                    ],
                    t2g_path=t2g_path,
                    ec_paths=[
                        filtered_results[prefix].get('ec')
                        for prefix in prefixes
                    ],
                    txnames_path=bus_result['txnames'],
                    loom=loom,
                    h5ad=h5ad,
                    by_name=by_name,
                    tcc=tcc,
                    nucleus=nucleus,
                    threads=threads,
                )
            )

    STATS.end()
    stats_path = STATS.save(os.path.join(out_dir, KB_INFO_FILENAME))
    results.update({'stats': stats_path})

    # Reports
    nb_path = os.path.join(out_dir, REPORT_NOTEBOOK_FILENAME)
    html_path = os.path.join(out_dir, REPORT_HTML_FILENAME)
    if report:
        logger.info(
            f'Writing report Jupyter notebook at {nb_path} and rendering it to {html_path}'
        )
        report_result = render_report(
            stats_path,
            bus_result['info'],
            unfiltered_results['inspect'],
            nb_path,
            html_path,
            temp_dir=temp_dir
        )

        unfiltered_results.update(report_result)

        for prefix in prefix_to_t2c:
            nb_path = os.path.join(
                out_dir, update_filename(REPORT_NOTEBOOK_FILENAME, prefix)
            )
            html_path = os.path.join(
                out_dir, update_filename(REPORT_HTML_FILENAME, prefix)
            )
            logger.info(
                f'Writing report Jupyter notebook at {nb_path} and rendering it to {html_path}'
            )
            report_result = render_report(
                stats_path,
                bus_result['info'],
                unfiltered_results[prefix]['inspect'],
                nb_path,
                html_path,
                unfiltered_results[prefix]['mtx'],
                unfiltered_results[prefix].get('barcodes'),
                unfiltered_results[prefix].get('genes'),
                t2g_path,
                temp_dir=temp_dir
            )
            unfiltered_results[prefix].update(report_result)
        if tcc:
            logger.warning(
                'Plots for TCC matrices have not yet been implemented. The HTML report will not contain any plots.'
            )

    return results


@logger.namespaced('count_velocity_smartseq3')
def count_velocity_smartseq3(
    index_path: str,
    t2g_path: str,
    cdna_t2c_path: str,
    intron_t2c_path: str,
    out_dir: str,
    fastqs: List[str],
    whitelist_path: Optional[str] = None,
    tcc: bool = False,
    mm: bool = False,
    temp_dir: str = 'tmp',
    threads: int = 8,
    memory: str = '4G',
    overwrite: bool = False,
    loom: bool = False,
    h5ad: bool = False,
    by_name: bool = False,
    inspect: bool = True,
    strand: Optional[Literal['unstranded', 'forward', 'reverse']] = None,
) -> Dict[str, Union[str, Dict[str, str]]]:
    """Generates count matrices for Smartseq3 (DEPRECATED).

    Args:
        index_path: Path to kallisto index
        t2g_path: Path to transcript-to-gene mapping
        out_dir: Path to output directory
        fastqs: List of FASTQ file paths
        whitelist_path: Path to whitelist, defaults to `None`
        tcc: Whether to generate a TCC matrix instead of a gene count matrix,
            defaults to `False`
        mm: Whether to include BUS records that pseudoalign to multiple genes,
            defaults to `False`
        temp_dir: Path to temporary directory, defaults to `tmp`
        threads: Pumber of threads to use, defaults to `8`
        memory: Amount of memory to use, defaults to `4G`
        overwrite: Overwrite an existing index file, defaults to `False`
        loom: Whether to convert the final count matrix into a loom file,
            defaults to `False`
        h5ad: Whether to convert the final count matrix into a h5ad file,
            defaults to `False`
        by_name: Aggregate counts by name instead of ID. Only affects when
            `tcc=False`.
        inspect: Whether or not to inspect the output BUS file and generate
            the inspect.json
        strand: Strandedness, defaults to `None`

    Returns:
        Dictionary containing paths to generated files
    """
    STATS.start()
    is_batch = isinstance(fastqs, str)
    BUS_CDNA_PREFIX = 'spliced'
    BUS_INTRON_PREFIX = 'unspliced'

    results = {}
    make_directory(out_dir)
    unfiltered_results = results.setdefault('unfiltered', {})

    bus_result = {
        'bus': os.path.join(out_dir, BUS_FILENAME),
        'ecmap': os.path.join(out_dir, ECMAP_FILENAME),
        'txnames': os.path.join(out_dir, TXNAMES_FILENAME),
        'info': os.path.join(out_dir, KALLISTO_INFO_FILENAME),
        'flens': os.path.join(out_dir, FLENS_FILENAME),
        'saved_index': os.path.join(out_dir, SAVED_INDEX_FILENAME)
    }
    if any(not os.path.exists(path)
           for name, path in bus_result.items()) or overwrite:
        # Pipe any remote files.
        fastqs = stream_batch(
            fastqs, temp_dir=temp_dir
        ) if is_batch else stream_fastqs(
            fastqs, temp_dir=temp_dir
        )
        bus_result = kallisto_bus(
            fastqs,
            index_path,
            'SMARTSEQ3',
            out_dir,
            threads=threads,
            paired=True,
            strand=strand,
        )
    else:
        logger.info(
            'Skipping kallisto bus because output files already exist. Use the --overwrite flag to overwrite.'
        )
    unfiltered_results.update(bus_result)

    sort_result = bustools_sort(
        bus_result['bus'],
        os.path.join(
            temp_dir,
            update_filename(os.path.basename(bus_result['bus']), SORT_CODE)
        ),
        temp_dir=temp_dir,
        threads=threads,
        memory=memory
    )
    logger.info('Whitelist not provided')
    whitelist_path = copy_or_create_whitelist(
        'SMARTSEQ3', sort_result['bus'], out_dir
    )
    unfiltered_results.update({'whitelist': whitelist_path})

    prev_result = sort_result
    if inspect:
        inspect_result = bustools_inspect(
            prev_result['bus'],
            os.path.join(out_dir, INSPECT_FILENAME),
            whitelist_path=whitelist_path,
        )
        unfiltered_results.update(inspect_result)
    prev_result = bustools_correct(
        prev_result['bus'],
        os.path.join(
            temp_dir,
            update_filename(os.path.basename(prev_result['bus']), CORRECT_CODE)
        ), whitelist_path
    )
    prev_result = bustools_sort(
        prev_result['bus'],
        os.path.join(out_dir, f'output.{UNFILTERED_CODE}.bus'),
        temp_dir=temp_dir,
        threads=threads,
        memory=memory
    )
    unfiltered_results.update({'bus_scs': prev_result['bus']})

    # Helper function to update results with suffix
    def update_results_with_suffix(current_results, new_results, suffix):
        current_results.update({
            f'{key}{suffix}': value
            for key, value in new_results.items()
        })

    # Write capture file & capture internal/umi records.
    capture_path = write_smartseq3_capture(
        os.path.join(out_dir, CAPTURE_FILENAME)
    )

    prefixes = [BUS_CDNA_PREFIX, BUS_INTRON_PREFIX]
    # The prefix and t2cs are swapped because we call bustools capture with
    # the --complement flag.
    prefix_to_t2c = {
        BUS_CDNA_PREFIX: intron_t2c_path,
        BUS_INTRON_PREFIX: cdna_t2c_path,
    }
    suffix_to_inspect_filename = {
        INTERNAL_SUFFIX: INSPECT_INTERNAL_FILENAME,
        UMI_SUFFIX: INSPECT_UMI_FILENAME,
    }
    for suffix, inspect_filename in suffix_to_inspect_filename.items():
        capture_result = bustools_capture(
            prev_result['bus'],
            os.path.join(out_dir, f'output{suffix}.bus'),
            capture_path,
            capture_type='umis',
            complement=suffix == UMI_SUFFIX
        )
        update_results_with_suffix(unfiltered_results, capture_result, suffix)

        if inspect:
            inspect_result = bustools_inspect(
                capture_result['bus'],
                os.path.join(out_dir, inspect_filename),
                whitelist_path=whitelist_path,
            )
            update_results_with_suffix(
                unfiltered_results, inspect_result, suffix
            )

        counts_dir = os.path.join(out_dir, f'{UNFILTERED_COUNTS_DIR}{suffix}')
        make_directory(counts_dir)
        if tcc:
            quant_dir = os.path.join(out_dir, f'{UNFILTERED_QUANT_DIR}{suffix}')
            make_directory(quant_dir)
        for prefix, t2c_path in prefix_to_t2c.items():
            prefix_capture_result = bustools_capture(
                capture_result['bus'],
                os.path.join(temp_dir, f'{prefix}{suffix}.bus'), t2c_path,
                bus_result['ecmap'], bus_result['txnames']
            )
            sort_result = bustools_sort(
                prefix_capture_result['bus'],
                os.path.join(
                    out_dir, f'{prefix}{suffix}.{UNFILTERED_CODE}.bus'
                ),
                temp_dir=temp_dir,
                threads=threads,
                memory=memory
            )
            prefix_results = unfiltered_results.setdefault(prefix, {})
            update_results_with_suffix(prefix_results, sort_result, suffix)

            if inspect:
                inpsect_result = bustools_inspect(
                    sort_result['bus'],
                    os.path.join(
                        out_dir, update_filename(inspect_filename, prefix)
                    ),
                )
                update_results_with_suffix(
                    prefix_results, inpsect_result, suffix
                )

            counts_prefix = os.path.join(counts_dir, prefix)
            count_result = bustools_count(
                sort_result['bus'],
                counts_prefix,
                t2g_path,
                bus_result['ecmap'],
                bus_result['txnames'],
                tcc=tcc,
                mm=mm or tcc,
                cm=suffix == INTERNAL_SUFFIX,
                umi_gene=suffix == UMI_SUFFIX,
            )
            update_results_with_suffix(prefix_results, count_result, suffix)

            if tcc:
                quant_result = kallisto_quant_tcc(
                    count_result['mtx'],
                    index_path,
                    bus_result['ecmap'],
                    t2g_path,
                    quant_dir,
                    flens_path=bus_result['flens'],
                    threads=threads,
                )
                update_results_with_suffix(prefix_results, quant_result, suffix)

        # After internal/UMI is done, create anndata separately for each
        if loom or h5ad:
            name = GENE_NAME
            if tcc:
                name = TRANSCRIPT_NAME

            convert_result = convert_matrices(
                quant_dir if tcc else counts_dir,
                [
                    unfiltered_results[prefix][f'mtx{suffix}']
                    for prefix in prefixes
                ],
                [
                    unfiltered_results[prefix][f'barcodes{suffix}']
                    for prefix in prefixes
                ],
                genes_paths=[
                    unfiltered_results[prefix][f'ec{suffix}'] if tcc else
                    unfiltered_results[prefix].get(f'genes{suffix}')
                    for prefix in prefixes
                ],
                t2g_path=t2g_path,
                ec_paths=[
                    unfiltered_results[prefix].get(f'ec{suffix}')
                    for prefix in prefixes
                ],
                txnames_path=bus_result['txnames'],
                name=name,
                loom=loom,
                h5ad=h5ad,
                by_name=by_name,
                tcc=False,
                threads=threads,
            )
            update_results_with_suffix(
                unfiltered_results, convert_result, suffix
            )

    STATS.end()
    stats_path = STATS.save(os.path.join(out_dir, KB_INFO_FILENAME))
    results.update({'stats': stats_path})
    return results
