import concurrent.futures
import functools
import os
import queue
import re
import shutil
import subprocess as sp
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.request import urlretrieve

import anndata
import ngs_tools as ngs
import pandas as pd
import scipy.io
from scipy import sparse

from .config import (
    get_bustools_binary_path,
    get_kallisto_binary_path,
    PLATFORM,
    TECHNOLOGIES_MAPPING,
    UnsupportedOSError,
)
from .dry import dryable
from .dry import utils as dry_utils
from .logging import logger
from .stats import STATS

TECHNOLOGY_PARSER = re.compile(r'^(?P<name>\S+)')
VERSION_PARSER = re.compile(r'^\S*? ([0-9]+).([0-9]+).([0-9]+)')

# These functions have been moved as of 0.26.1 to the ngs_tools library but are
# imported from this file in other places. For now, let's keep these here.
# TODO: remove these
open_as_text = ngs.utils.open_as_text
decompress_gzip = ngs.utils.decompress_gzip
compress_gzip = ngs.utils.compress_gzip
concatenate_files = ngs.utils.concatenate_files_as_text
download_file = ngs.utils.download_file
get_temporary_filename = dryable(dry_utils.get_temporary_filename)(
    ngs.utils.mkstemp
)


def update_filename(filename: str, code: str) -> str:
    """Update the provided path with the specified code.

    For instance, if the `path` is 'output.bus' and `code` is `s` (for sort),
    this function returns `output.s.bus`.

    Args:
        filename: filename (NOT path)
        code: code to append to filename

    Returns:
        Path updated with provided code
    """
    name, extension = os.path.splitext(filename)
    return f'{name}.{code}{extension}'


@dryable(dry_utils.make_directory)
def make_directory(path: str):
    """Quietly make the specified directory (and any subdirectories).

    This function is a wrapper around os.makedirs. It is used so that
    the appropriate mkdir command can be printed for dry runs.

    Args:
        path: Path to directory to make
    """
    os.makedirs(path, exist_ok=True)


@dryable(dry_utils.remove_directory)
def remove_directory(path: str):
    """Quietly make the specified directory (and any subdirectories).

    This function is a wrapper around shutil.rmtree. It is used so that
    the appropriate rm command can be printed for dry runs.

    Args:
        path: Path to directory to remove
    """
    shutil.rmtree(path, ignore_errors=True)


@dryable(dry_utils.run_executable)
def run_executable(
    command: List[str],
    stdin: Optional[int] = None,
    stdout: int = sp.PIPE,
    stderr: int = sp.PIPE,
    wait: bool = True,
    stream: bool = True,
    quiet: bool = False,
    returncode: int = 0,
    alias: bool = True,
    record: bool = True,
) -> Union[Tuple[sp.Popen, str, str], sp.Popen]:
    """Execute a single shell command.

    Args:
        command: A list representing a single shell command
        stdin: Object to pass into the `stdin` argument for `subprocess.Popen`,
            defaults to `None`
        stdout: Object to pass into the `stdout` argument for `subprocess.Popen`,
            defaults to `subprocess.PIPE`
        stderr: Object to pass into the `stderr` argument for `subprocess.Popen`,
            defaults to `subprocess.PIPE`
        wait: Whether to wait until the command has finished, defaults to `True`
        stream: Whether to stream the output to the command line, defaults to `True`
        quiet: Whether to not display anything to the command line and not check the return code,
            defaults to `False`
        returncode: The return code expected if the command runs as intended,
            defaults to `0`
        alias: Whether to use the basename of the first element of `command`,
            defaults to `True`
        record: Whether to record the call statistics, defaults to `True`

    Returns:
        (the spawned process, list of strings printed to stdout,
            list of strings printed to stderr) if `wait=True`.
            Otherwise, the spawned process
    """
    command = [str(c) for c in command]
    c = command.copy()
    if alias:
        c[0] = os.path.basename(c[0])
    if not quiet:
        logger.debug(' '.join(c))
    if not wait and record:
        STATS.command(c)
    start = time.time()
    p = sp.Popen(
        command,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        universal_newlines=wait,
        bufsize=1 if wait else -1,
    )

    # Helper function to read from a pipe and put the output to a queue.
    def reader(pipe, qu, stop_event, name):
        while not stop_event.is_set():
            for _line in pipe:
                line = _line.strip()
                qu.put((name, line))

    # Wait if desired.
    if wait:
        stdout = ''
        stderr = ''
        out = []
        out_queue = queue.Queue()
        stop_event = threading.Event()
        stdout_reader = threading.Thread(
            target=reader,
            args=(p.stdout, out_queue, stop_event, 'stdout'),
            daemon=True
        )
        stderr_reader = threading.Thread(
            target=reader,
            args=(p.stderr, out_queue, stop_event, 'stderr'),
            daemon=True
        )
        stdout_reader.start()
        stderr_reader.start()

        while p.poll() is None:
            while not out_queue.empty():
                name, line = out_queue.get()
                if stream and not quiet:
                    logger.debug(line)
                out.append(line)
                if name == 'stdout':
                    stdout += f'{line}\n'
                elif name == 'stderr':
                    stderr += f'{line}\n'
            else:
                time.sleep(0.1)

        # Stop readers & flush queue
        stop_event.set()
        time.sleep(1)
        while not out_queue.empty():
            name, line = out_queue.get()
            if stream and not quiet:
                logger.debug(line)
            out.append(line)
            if name == 'stdout':
                stdout += f'{line}\n'
            elif name == 'stderr':
                stderr += f'{line}\n'
        if record:
            STATS.command(c, runtime=time.time() - start)

        if not quiet and p.returncode != returncode:
            logger.error('\n'.join(out))
            raise sp.CalledProcessError(p.returncode, ' '.join(command))
        # logger.info(stdout)

    return (p, stdout, stderr) if wait else p


def get_kallisto_version() -> Optional[Tuple[int, int, int]]:
    """Get the provided Kallisto version.

    This function parses the help text by executing the included Kallisto binary.

    Returns:
        Major, minor, patch versions
    """
    p, stdout, stderr = run_executable([get_kallisto_binary_path()],
                                       quiet=True,
                                       returncode=1,
                                       record=False)
    match = VERSION_PARSER.match(stdout)
    return tuple(int(ver) for ver in match.groups()) if match else None


def get_bustools_version() -> Optional[Tuple[int, int, int]]:
    """Get the provided Bustools version.

    This function parses the help text by executing the included Bustools binary.

    Returns:
        Major, minor, patch versions
    """
    p, stdout, stderr = run_executable([get_bustools_binary_path()],
                                       quiet=True,
                                       returncode=1,
                                       record=False)
    match = VERSION_PARSER.match(stdout)
    return tuple(int(ver) for ver in match.groups()) if match else None


def parse_technologies(lines: List[str]) -> Set[str]:
    """Parse a list of strings into a list of supported technologies.

    This function parses the technologies printed by running `kallisto bus --list`.

    Args:
        lines: The output of `kallisto bus --list` split into lines

    Returns:
        Set of technologies
    """
    parsing = False
    technologies = set()
    for line in lines:
        if line.startswith('-'):
            parsing = True
            continue

        if parsing:
            if line.isspace():
                break
            match = TECHNOLOGY_PARSER.match(line)
            if match:
                technologies.add(match['name'])
    return technologies


def get_supported_technologies() -> Set[str]:
    """Runs 'kallisto bus --list' to fetch a list of supported technologies.

    Returns:
        Set of technologies
    """
    p, stdout, stderr = run_executable([
        get_kallisto_binary_path(), 'bus', '--list'
    ],
                                       quiet=True,
                                       returncode=1,
                                       record=False)
    return parse_technologies(stdout)


def whitelist_provided(technology: str) -> bool:
    """Determine whether or not the whitelist for a technology is provided.

    Args:
        technology: The name of the technology

    Returns:
        Whether the whitelist is provided
    """
    upper = technology.upper()
    return upper in TECHNOLOGIES_MAPPING and TECHNOLOGIES_MAPPING[
        upper].chemistry.has_whitelist


@dryable(dry_utils.move_file)
def move_file(source: str, destination: str) -> str:
    """Move a file from source to destination, overwriting the file if the
    destination exists.

    Args:
        source: Path to source file
        destination: Path to destination

    Returns:
        Path to moved file
    """
    shutil.move(source, destination)
    return destination


@dryable(dry_utils.copy_whitelist)
def copy_whitelist(technology: str, out_dir: str) -> str:
    """Copies provided whitelist for specified technology.

    Args:
        technology: The name of the technology
        out_dir: Directory to put the whitelist

    Returns:
        Path to whitelist
    """
    technology = TECHNOLOGIES_MAPPING[technology.upper()]
    archive_path = technology.chemistry.whitelist_path
    whitelist_path = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(archive_path))[0]
    )
    with open_as_text(archive_path, 'r') as f, open(whitelist_path, 'w') as out:
        out.write(f.read())
    return whitelist_path


@dryable(dry_utils.create_10x_feature_barcode_map)
def create_10x_feature_barcode_map(out_path: str) -> str:
    """Create a feature-barcode map for the 10x Feature Barcoding technology.

    Args:
        out_path: Path to the output mapping file

    Returns:
        Path to map
    """
    chemistry = ngs.chemistry.get_chemistry('10xFB')
    gex = chemistry.chemistry('gex')
    fb = chemistry.chemistry('fb')
    with open_as_text(fb.whitelist_path, 'r') as fb_f, open_as_text(
            gex.whitelist_path, 'r') as gex_f, open(out_path, 'w') as out:
        for fb_line, gex_line in zip(fb_f, gex_f):
            fb_barcode = fb_line.strip()
            gex_barcode = gex_line.strip()
            out.write(f'{fb_barcode}\t{gex_barcode}\n')
    return out_path


@dryable(dry_utils.stream_file)
def stream_file(url: str, path: str) -> str:
    """Creates a FIFO file to use for piping remote files into processes.

    This function spawns a new thread to download the remote file into a FIFO
    file object. FIFO file objects are only supported on unix systems.

    Args:
        url: Url to the file
        path: Path to place FIFO file

    Returns:
        Path to FIFO file

    Raises:
        UnsupportedOSError: If the OS is Windows
    """
    # Windows does not support FIFO files.
    if PLATFORM == 'windows':
        raise UnsupportedOSError((
            'Windows does not support piping remote files.'
            'Please download the file manually.'
        ))
    else:
        logger.info('Piping {} to {}'.format(url, path))
        os.mkfifo(path)
        t = threading.Thread(target=urlretrieve, args=(url, path), daemon=True)
        t.start()
    return path


def read_t2g(t2g_path: str) -> Dict[str, Tuple[str, ...]]:
    """Given a transcript-to-gene mapping path, read it into a dictionary.
    The first column is always assumed to tbe the transcript IDs.

    Args:
        t2g_path: Path to t2g

    Returns:
        Dictionary containing transcript IDs as keys and all other columns
            as a tuple as values
    """
    t2g = {}
    with open_as_text(t2g_path, 'r') as f:
        for line in f:
            if line.isspace():
                continue
            split = line.strip().split('\t')
            transcript = split[0]
            other = tuple(split[1:])
            if transcript in t2g:
                logger.warning(
                    f'Found duplicate entries for {transcript} in {t2g_path}. '
                    'Earlier entries will be ignored.'
                )
            t2g[transcript] = other
    return t2g


def obtain_gene_names(
    t2g_path: str,
    gene_names_list: Union[str, List[str]],
    verbose: Optional[bool] = True,
    clean_dups: Optional[bool] = True
) -> List[str]:
    """Given a transcript-to-gene mapping path and list of gene IDs,
    return a list of cleaned-up gene names (wherein blank names are simply
    replaced by the corresponding gene ID, as are duplicate names if specified)

    Args:
        t2g_path: Path to t2g
        gene_names_list: List of gene IDs or path to list of gene IDs
        verbose: Whether to warn about the number of blank names, defaults to `True`
        clean_dups: Whether to convert duplicate names to gene IDs, defaults to `True`

    Returns:
        List of gene names
    """
    is_geneid_path = isinstance(gene_names_list, str)
    var_names = []
    if is_geneid_path:
        if not os.path.exists(gene_names_list):
            return []
        with open_as_text(gene_names_list, 'r') as f:
            var_names = [line.strip() for line in f]
    else:
        var_names = gene_names_list

    t2g = read_t2g(t2g_path)
    id_to_name = {}
    for transcript, attributes in t2g.items():
        if len(attributes) > 1:
            id_to_name[attributes[0]] = attributes[1]
    # Locate duplicates:
    names_set = set([])
    duplicates_set = set([])
    if clean_dups:
        for gene_id in var_names:
            if id_to_name.get(gene_id):
                if id_to_name[gene_id] in names_set:
                    duplicates_set.add(id_to_name[gene_id])
                names_set.add(id_to_name[gene_id])
    # Now make list of cleaned-up gene names:
    gene_names = []
    n_no_name = 0
    for gene_id in var_names:
        if id_to_name.get(gene_id) and not (id_to_name[gene_id]
                                            in duplicates_set):
            gene_names.append(id_to_name[gene_id])
        else:  # blank names and duplicate names are considered missing
            gene_names.append(gene_id)
            n_no_name += 1
    if n_no_name > 0 and verbose:
        logger.warning(
            f'{n_no_name} gene IDs do not have corresponding valid gene names. '
            'These genes will use their gene IDs instead.'
        )
    return gene_names


def write_list_to_file(strings: List[str], str_path: str) -> str:
    """Write out a list of strings.

    Args:
        strings: List of strings to output
        str_path: Path to output

    Returns:
        Path to written file
    """
    with open_as_text(str_path, 'w') as out:
        for s in strings:
            out.write(f'{s}\n')
    return str_path


def collapse_anndata(
    adata: anndata.AnnData, by: Optional[str] = None
) -> anndata.AnnData:
    """Collapse the given Anndata by summing duplicate rows. The `by` argument
    specifies which column to use. If not provided, the index is used.

    Note:
        This function also collapses any existing layers. Additionally, the
        returned AnnData will have the values used to collapse as the index.

    Args:
        adata: The Anndata to collapse
        by: The column to collapse by. If not provided, the index is used. When
            this column contains missing values (i.e. nan or None), these
            columns are removed.

    Returns:
        A new collapsed Anndata object. All matrices are sparse, regardless of
        whether or not they were in the input Anndata.
    """
    var = adata.var
    if by is not None:
        var = var.set_index(by)
    na_mask = var.index.isna()
    adata = adata[:, ~na_mask].copy()
    adata.var = var[~na_mask]

    if not any(adata.var.index.duplicated()):
        return adata

    var_indices = {}
    for i, index in enumerate(adata.var.index):
        var_indices.setdefault(index, []).append(i)

    # Convert all original matrices to csc for fast column operations
    X = sparse.csc_matrix(adata.X)
    layers = {
        layer: sparse.csc_matrix(adata.layers[layer])
        for layer in adata.layers
    }
    new_index = []
    # lil_matrix is efficient for row-by-row construction
    new_X = sparse.lil_matrix((len(var_indices), adata.shape[0]))
    new_layers = {layer: new_X.copy() for layer in adata.layers}
    for i, (index, indices) in enumerate(var_indices.items()):
        new_index.append(index)
        new_X[i] = X[:, indices].sum(axis=1).flatten()
        for layer in layers.keys():
            new_layers[layer][i] = layers[layer][:,
                                                 indices].sum(axis=1).flatten()

    return anndata.AnnData(
        X=new_X.T.tocsr(),
        layers={layer: new_layers[layer].T.tocsr()
                for layer in new_layers},
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Series(new_index, name=adata.var.index.name)),
    )


def import_tcc_matrix_as_anndata(
    matrix_path: str,
    barcodes_path: str,
    ec_path: str,
    txnames_path: str,
    threads: int = 8,
    loom: bool = False,
    loom_names: List[str] = None,
    batch_barcodes_path: Optional[str] = None,
) -> anndata.AnnData:
    """Import a TCC matrix as an Anndata object.

    Args:
        matrix_path: Path to the matrix ec file
        barcodes_path: Path to the barcodes txt file
        genes_path: Path to the ec txt file
        txnames_path: Path to transcripts.txt generated by `kallisto bus`
        threads: Number of threads, defaults to `8`
        loom: Whether to prepare anndata for loom file, defaults to `False`
        loom_names: Names for cols and rows in anndata, defaults to `None`
        batch_barcodes_path: Path to barcodes prefixed with sample ID,
            defaults to `None`

    Returns:
        A new Anndata object
    """
    name_column = 'transcript_ids' if not loom else loom_names[1]
    bc_name = 'barcode' if not loom else loom_names[0]
    df_barcodes = pd.read_csv(
        barcodes_path, index_col=0, header=None, names=[bc_name]
    )
    if (batch_barcodes_path):
        df_batch_barcodes = pd.read_csv(
            batch_barcodes_path, index_col=0, header=None, names=[bc_name]
        )
        df_barcodes.index = df_batch_barcodes.index + df_barcodes.index
    df_ec = pd.read_csv(
        ec_path,
        index_col=0,
        header=None,
        names=['ec', 'transcripts'],
        sep='\t',
        dtype=str
    )
    df_ec.index = df_ec.index.astype(str)  # To prevent logging from anndata
    with open(txnames_path, 'r') as f:
        transcripts = [
            line.strip() for line in f.readlines() if not line.strip().isspace()
        ]

    ts = list(df_ec.transcripts)
    get_transcript_ids = lambda ts, transcripts: [
        ';'.join(transcripts[int(i)] for i in t.split(',')) for t in ts
    ]
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        chunk = int(len(ts) / threads) + 1
        for i in range(threads):
            future = executor.submit(
                get_transcript_ids, ts[i * chunk:(i + 1) * chunk], transcripts
            )
            futures.append(future)
    transcript_ids = []
    for future in futures:
        transcript_ids += future.result()
    df_ec[name_column] = pd.Categorical(transcript_ids)
    df_ec.drop('transcripts', axis=1, inplace=True)
    return anndata.AnnData(
        X=scipy.io.mmread(matrix_path).tocsr(), obs=df_barcodes, var=df_ec
    )


def import_matrix_as_anndata(
    matrix_path: str,
    barcodes_path: str,
    genes_path: str,
    t2g_path: Optional[str] = None,
    name: str = 'gene',
    by_name: bool = False,
    loom: bool = False,
    loom_names: List[str] = None,
    batch_barcodes_path: Optional[str] = None,
) -> anndata.AnnData:
    """Import a matrix as an Anndata object.

    Args:
        matrix_path: Path to the matrix ec file
        barcodes_path: Path to the barcodes txt file
        genes_path: Path to the genes txt file
        t2g_path: Path to transcript-to-gene mapping. If this is provided,
            the third column of the mapping is appended to the anndata var,
            defaults to `None`
        name: Name of the columns, defaults to "gene"
        by_name: Aggregate counts by name instead of ID. `t2g_path` must be
            provided and contain names.
        loom: Whether to prepare anndata for loom file, defaults to `False`
        loom_names: Names for cols and rows in anndata, defaults to `None`
        batch_barcodes_path: Path to barcodes prefixed with sample ID,
            defaults to `None`

    Returns:
        A new Anndata object
    """
    name_column = f'{name}_id' if not loom else loom_names[1]
    bc_name = 'barcode' if not loom else loom_names[0]
    df_barcodes = pd.read_csv(
        barcodes_path, index_col=0, header=None, names=[bc_name]
    )
    if (batch_barcodes_path):
        df_batch_barcodes = pd.read_csv(
            batch_barcodes_path, index_col=0, header=None, names=[bc_name]
        )
        df_barcodes.index = df_batch_barcodes.index + df_barcodes.index
    df_genes = pd.read_csv(
        genes_path,
        header=None,
        index_col=0,
        names=[name_column],
        sep='\t',
        dtype={0: str}
    )
    mtx = scipy.io.mmread(matrix_path)
    adata = collapse_anndata(
        anndata.AnnData(X=mtx.tocsr(), obs=df_barcodes, var=df_genes)
    )

    if t2g_path and by_name:
        gene_names = obtain_gene_names(
            t2g_path, adata.var_names.to_list(), False
        )
        adata.var[name_column] = pd.Categorical(gene_names)

    return (
        collapse_anndata(adata, by=name_column)
        if name_column in adata.var.columns and by_name else adata
    )


def overlay_anndatas(
    adata_spliced: anndata.AnnData,
    adata_unspliced: anndata.AnnData,
    adata_ambiguous: anndata.AnnData = None
) -> anndata.AnnData:
    """'Overlays' anndata objects by taking the intersection of the obs and var
    of each anndata.

    Note:
        Matrices generated by kallisto | bustools always contain all genes,
        even if they have zero counts. Therefore, taking the intersection
        is not entirely necessary but is done as a sanity check.

    Args:
        adata_spliced: An Anndata object
        adata_unspliced: An Anndata object
        adata_ambiguous: An Anndata object, default `None`

    Returns:
        A new Anndata object
    """
    obs_idx = adata_spliced.obs.index.intersection(adata_unspliced.obs.index)
    var_idx = adata_spliced.var.index.intersection(adata_unspliced.var.index)
    spliced_intersection = adata_spliced[obs_idx][:, var_idx]
    unspliced_intersection = adata_unspliced[obs_idx][:, var_idx]
    a_layers = {
        'mature': spliced_intersection.X,
        'nascent': unspliced_intersection.X
    }
    sum_X = spliced_intersection.X + unspliced_intersection.X
    ambiguous_intersection = None
    if adata_ambiguous is not None:
        ambiguous_intersection = adata_ambiguous[obs_idx][:, var_idx]
        a_layers.update({'ambiguous': ambiguous_intersection.X})
        sum_X = sum_X + ambiguous_intersection.X

    df_obs = unspliced_intersection.obs
    df_var = unspliced_intersection.var
    return anndata.AnnData(X=sum_X, layers=a_layers, obs=df_obs, var=df_var)


def sum_anndatas(
    adata_spliced: anndata.AnnData, adata_unspliced: anndata.AnnData
) -> anndata.AnnData:
    """Sum the counts in two anndata objects by taking the intersection of
    both matrices and adding the values together.

    Note:
        Matrices generated by kallisto | bustools always contain all genes,
        even if they have zero counts. Therefore, taking the intersection
        is not entirely necessary but is done as a sanity check.

    Args:
        adata_spliced: An Anndata object
        adata_unspliced: An Anndata object

    Returns:
        A new Anndata object
    """
    obs_idx = adata_spliced.obs.index.intersection(adata_unspliced.obs.index)
    var_idx = adata_spliced.var.index.intersection(adata_unspliced.var.index)
    spliced_intersection = adata_spliced[obs_idx][:, var_idx]
    unspliced_intersection = adata_unspliced[obs_idx][:, var_idx]

    df_obs = unspliced_intersection.obs
    df_var = unspliced_intersection.var
    return anndata.AnnData(
        X=spliced_intersection.X + unspliced_intersection.X,
        obs=df_obs,
        var=df_var
    )


def do_sum_matrices(
    mtx1_path, mtx2_path, out_path, mm=False, header_line=None
) -> str:
    """Sums up two matrices given two matrix files.

    Args:
        mtx1_path: First matrix file path
        mtx2_path: Second matrix file path
        out_path: Output file path
        mm: Whether to allow multimapping (i.e. decimals)
        header_line: The header line if we have it

    Returns:
        Output file path
    """
    logger.info('Summing matrices into {}'.format(out_path))
    n = 0
    header = []
    with open_as_text(mtx1_path,
                      'r') as f1, open_as_text(mtx2_path,
                                               'r') as f2, open(out_path,
                                                                'w') as out:
        eof1 = eof2 = pause1 = pause2 = False
        nums = [0, 0, 0]
        nums1 = nums2 = to_write = None
        if header_line:
            out.write("%%MatrixMarket matrix coordinate real general\n%\n")
        while not eof1 or not eof2:
            s1 = f1.readline() if not eof1 and not pause1 else '%'
            s2 = f2.readline() if not eof2 and not pause2 else '%'
            if not s1:
                pause1 = eof1 = True
            if not s2:
                pause2 = eof2 = True
            _nums1 = _nums2 = []
            if not eof1 and s1[0] != '%':
                _nums1 = s1.split()
                if not mm:
                    _nums1[0] = int(_nums1[0])
                    _nums1[1] = int(_nums1[1])
                    _nums1[2] = int(float(_nums1[2]))
                else:
                    _nums1[0] = int(_nums1[0])
                    _nums1[1] = int(_nums1[1])
                    _nums1[2] = float(_nums1[2])
            if not eof2 and s2[0] != '%':
                _nums2 = s2.split()
                if not mm:
                    _nums2[0] = int(_nums2[0])
                    _nums2[1] = int(_nums2[1])
                    _nums2[2] = int(float(_nums2[2]))
                else:
                    _nums2[0] = int(_nums2[0])
                    _nums2[1] = int(_nums2[1])
                    _nums2[2] = float(_nums2[2])
            if nums1 is not None:
                _nums1 = nums1
                nums1 = None
            if nums2 is not None:
                _nums2 = nums2
                nums2 = None
            if eof1 and eof2:
                # Both mtxs are done
                break
            elif eof1:
                # mtx1 is done
                nums = _nums2
                pause2 = False
            elif eof2:
                # mtx2 is done
                nums = _nums1
                pause1 = False
            elif eof1 and eof2:
                # Both mtxs are done
                break
            # elif (len(_nums1) != len(_nums2)):
            #    # We have a problem
            #    raise Exception("Summing up two matrix files failed")
            elif not _nums1 or not _nums2:
                # We have something other than a matrix line
                continue
            elif not header:
                # We are at the header line and need to read it in
                if (_nums1[0] != _nums2[0] or _nums1[1] != _nums2[1]):
                    raise Exception(
                        "Summing up two matrix files failed: Headers incompatible"
                    )
                else:
                    header = [_nums1[0], _nums1[1]]
                if header_line:
                    out.write(header_line)
                continue
            elif (_nums1[0] > _nums2[0]
                  or (_nums1[0] == _nums2[0] and _nums1[1] > _nums2[1])):
                # If we're further in mtx1 than mtx2
                nums = _nums2
                pause1 = True
                pause2 = False
                nums1 = _nums1
                nums2 = None
            elif (_nums2[0] > _nums1[0]
                  or (_nums2[0] == _nums1[0] and _nums2[1] > _nums1[1])):
                # If we're further in mtx2 than mtx1
                nums = _nums1
                pause2 = True
                pause1 = False
                nums2 = _nums2
                nums1 = None
            elif _nums1[0] == _nums2[0] and _nums1[1] == _nums2[1]:
                # If we're at the same location in mtx1 and mtx2
                nums = _nums1
                nums[2] += _nums2[2]
                pause1 = pause2 = False
                nums1 = nums2 = None
            else:
                # Shouldn't happen
                raise Exception(
                    "Summing up two matrix files failed: Assertion failed"
                )
            # Write out a line
            _nums_prev = to_write
            if (_nums_prev and _nums_prev[0] == nums[0]
                    and _nums_prev[1] == nums[1]):
                nums[2] += _nums_prev[2]
                pause1 = pause2 = False
                to_write = [nums[0], nums[1], nums[2]]
            else:
                if to_write:
                    if header_line:
                        if mm and to_write[2].is_integer():
                            to_write[2] = int(to_write[2])
                        out.write(
                            f'{to_write[0]} {to_write[1]} {to_write[2]}\n'
                        )
                    n += 1
                to_write = [nums[0], nums[1], nums[2]]
        if to_write:
            if header_line:
                if mm and to_write[2].is_integer():
                    to_write[2] = int(to_write[2])
                out.write(f'{to_write[0]} {to_write[1]} {to_write[2]}\n')
            n += 1
    if not header_line:
        header_line = f'{header[0]} {header[1]} {n}\n'
        do_sum_matrices(mtx1_path, mtx2_path, out_path, mm, header_line)
    return out_path


def restore_cwd(func: Callable) -> Callable:
    """Function decorator to decorate functions that change the current working
    directory. When such a function is decorated with this function, the
    current working directory is restored to its previous state when the
    function exits.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        old_cwd = os.path.abspath(os.getcwd())
        try:
            return func(*args, **kwargs)
        finally:
            os.chdir(old_cwd)

    return wrapper
