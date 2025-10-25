import glob
import itertools
import os
import tarfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import ngs_tools as ngs
import pandas as pd

from .config import get_kallisto_binary_path
from .logging import logger
from .utils import (
    concatenate_files,
    decompress_gzip,
    download_file,
    get_temporary_filename,
    open_as_text,
    run_executable,
)


class RefError(Exception):
    pass


def generate_kite_fasta(
    feature_path: str,
    out_path: str,
    no_mismatches: bool = False
) -> Tuple[str, int]:
    """Generate a FASTA file for feature barcoding with the KITE workflow.

    This FASTA contains all sequences that are 1 hamming distance from the
    provided barcodes. The file of barcodes must be a 2-column TSV containing
    the barcode sequences in the first column and their corresponding feature
    name in the second column. If hamming distance 1 variants collide for any
    pair of barcodes, the hamming distance 1 variants for those barcodes are
    not generated.

    Args:
        feature_path: Path to TSV containing barcodes and feature names
        out_path: Path to FASTA to generate
        no_mismatches: Whether to generate hamming distance 1 variants,
            defaults to `False`

    Returns:
        Path to generated FASTA, smallest barcode length

    Raises:
        RefError: If there are barcodes of different lengths or if there are
            duplicate barcodes
    """

    def generate_mismatches(name, sequence):
        """Helper function to generate 1 hamming distance mismatches.
        """
        sequence = sequence.upper()
        for i in range(len(sequence)):
            base = sequence[i]
            before = sequence[:i]
            after = sequence[i + 1:]

            for j, different in enumerate([b for b in ['A', 'C', 'G', 'T']
                                           if b != base]):
                yield f'{name}-{i}.{j+1}', f'{before}{different}{after}'

    df_features = pd.read_csv(
        feature_path, sep='\t', header=None, names=['sequence', 'name']
    )

    lengths = set()
    features = {}
    variants = {}
    # Generate all feature barcode variations before saving to check for collisions.
    for i, row in df_features.iterrows():
        # Check that the first column contains the sequence
        # and the second column the feature name.
        if ngs.sequence.SEQUENCE_PARSER.search(row.sequence.upper()):
            raise RefError((
                f'Encountered non-ATCG basepairs in barcode sequence {row.sequence}. '
                'Does the first column contain the sequences and the second column the feature names?'
            ))

        lengths.add(len(row.sequence))
        features[row['name']] = row.sequence
        variants[row['name']] = {
            name: seq
            for name, seq in generate_mismatches(row['name'], row.sequence)
            if not no_mismatches
        }

    # Check duplicate barcodes.
    duplicates = set([
        bc for bc in features.values() if list(features.values()).count(bc) > 1
    ])
    if len(duplicates) > 0:
        raise RefError(
            'Duplicate feature barcodes: {}'.format(' '.join(duplicates))
        )
    if len(lengths) > 1:
        logger.warning(
            'Detected barcodes of different lengths: {}'.format(
                ','.join(str(l) for l in lengths)  # noqa
            )
        )
    # Find & remove collisions between barcode and variants
    for feature in variants.keys():
        _variants = variants[feature]
        collisions = set(_variants.values()) & set(features.values())
        if collisions:
            # Remove collisions
            logger.warning(
                f'Colision detected between variants of feature barcode {feature} '
                'and feature barcode(s). These variants will be removed.'
            )
            variants[feature] = {
                name: seq
                for name, seq in _variants.items()
                if seq not in collisions
            }

    # Find & remove collisions between variants
    for f1, f2 in itertools.combinations(variants.keys(), 2):
        v1 = variants[f1]
        v2 = variants[f2]

        collisions = set(v1.values()) & set(v2.values())
        if collisions:
            logger.warning(
                f'Collision(s) detected between variants of feature barcodes {f1} and {f2}: '
                f'{",".join(collisions)}. These variants will be removed.'
            )

            # Remove collisions
            variants[f1] = {
                name: seq
                for name, seq in v1.items()
                if seq not in collisions
            }
            variants[f2] = {
                name: seq
                for name, seq in v2.items()
                if seq not in collisions
            }

    # Write FASTA
    with ngs.fasta.Fasta(out_path, 'w') as f:
        for feature, barcode in features.items():
            attributes = {'feature_id': feature}
            header = ngs.fasta.FastaEntry.make_header(feature, attributes)
            entry = ngs.fasta.FastaEntry(header, barcode)
            f.write(entry)

            for name, variant in variants[feature].items():
                header = ngs.fasta.FastaEntry.make_header(name, attributes)
                entry = ngs.fasta.FastaEntry(header, variant)
                f.write(entry)

    return out_path, min(lengths)


def create_t2g_from_fasta(
    fasta_path: str, t2g_path: str, aa_flag: bool = False
) -> Dict[str, str]:
    """Parse FASTA headers to get transcripts-to-gene mapping.

    Args:
        fasta_path: Path to FASTA file
        t2g_path: Path to output transcript-to-gene mapping

    Returns:
        Dictionary containing path to generated t2g mapping
    """
    logger.info(f'Creating transcript-to-gene mapping at {t2g_path}')

    if aa_flag:
        with open(fasta_path, 'r') as f_in, open_as_text(t2g_path,
                                                         'w') as f_out:
            fasta_lines = f_in.readlines()
            for line in fasta_lines:
                if ">" in line:
                    label = line.split(">")[-1].split(" ")[0].replace("\n", "")
                    f_out.write(f'{label}\t{label}\n')

    else:
        with ngs.fasta.Fasta(fasta_path,
                             'r') as f_in, open_as_text(t2g_path, 'w') as f_out:
            for entry in f_in:
                attributes = entry.attributes

                if 'feature_id' in attributes:
                    feature_id = attributes['feature_id']
                    row = [entry.name, feature_id, feature_id]
                else:
                    gene_id = attributes['gene_id']
                    gene_name = attributes.get('gene_name', '')
                    transcript_name = attributes.get('transcript_name', '')
                    chromosome = attributes['chr']
                    start = attributes['start']
                    end = attributes['end']
                    strand = attributes['strand']
                    row = [
                        entry.name,
                        gene_id,
                        gene_name,
                        transcript_name,
                        chromosome,
                        start,
                        end,
                        strand,
                    ]
                f_out.write('\t'.join(str(item) for item in row) + '\n')

    return {'t2g': t2g_path}


def create_t2c(fasta_path: str, t2c_path: str) -> Dict[str, str]:
    """Creates a transcripts-to-capture list from a FASTA file.

    Args:
        fasta_path: Path to FASTA file
        t2c_path: Path to output transcripts-to-capture list

    Returns:
        Dictionary containing path to generated t2c list
    """
    with ngs.fasta.Fasta(fasta_path, 'r') as f_in, open_as_text(t2c_path,
                                                                'w') as f_out:
        for entry in f_in:
            f_out.write(f'{entry.name}\n')
    return {'t2c': t2c_path}


def kallisto_index(
    fasta_path: str,
    index_path: str,
    k: int = 31,
    threads: int = 8,
    dlist: str = None,
    dlist_overhang: int = 1,
    make_unique: bool = False,
    aa: bool = False,
    distinguish: bool = False,
    max_ec_size: int = None,
    temp_dir: str = 'tmp',
) -> Dict[str, str]:
    """Runs `kallisto index`.

    Args:
        fasta_path: path to FASTA file
        index_path: path to output kallisto index
        k: k-mer length, defaults to 31
        threads: Number of threads to use, defaults to `8`
        dlist: Path to a FASTA-file containing sequences to mask from quantification,
            defaults to `None`
        dlist_overhang: The overhang to use for the D-list, defaults to `1`
        make_unique: Replace repeated target names with unique names, defaults to `False`
        aa: Generate index from a FASTA-file containing amino acid sequences,
            defaults to `False`
        distinguish: Generate a color-based-on-target-name index,
            defaults to `False`
        max_ec_size: Sets max size of equivalence class, defaults to `None`

    Returns:
        Dictionary containing path to generated index
    """
    logger.info(f'Indexing {fasta_path} to {index_path}')
    command = [get_kallisto_binary_path(), 'index', '-i', index_path, '-k', k]
    if threads > 1:
        command += ['-t', threads]
    if dlist:
        command += ['-d', dlist]
    if make_unique:
        command += ['--make-unique']
    if aa:
        command += ['--aa']
    if distinguish:
        command += ['--distinguish']
    if max_ec_size:
        command += ['-e', max_ec_size]
    if dlist_overhang > 1:
        command += ['--d-list-overhang', dlist_overhang]
    if temp_dir != 'tmp':
        command += ['-T', temp_dir]
    if ',' in fasta_path:
        fasta_paths = fasta_path.split(',')
        for fp in fasta_paths:
            command += [fp]
    else:
        command += [fasta_path]
    run_executable(command)
    return {'index': index_path}


def get_dlist_fasta(fasta_path: str = None, temp_dir: str = 'tmp') -> str:
    """Downloads the D-list FASTA to temporary file if URL supplied

    Args:
        fasta_path: Path to FASTA file
        temp_dir: Path to temporary directory, defaults to `tmp`

    Returns:
        Path to D-list FASTA
    """

    if not fasta_path:
        return fasta_path
    if "://" not in fasta_path:  # Not a URL
        return fasta_path
    new_fasta_path = get_temporary_filename(temp_dir)
    fasta_path_array = [fasta_path]
    if fasta_path.count("://") > 1:
        fasta_path_array = fasta_path.split(",")
    logger.info(f'Extracting {fasta_path} into {new_fasta_path}')
    with ngs.fasta.Fasta(new_fasta_path, 'w') as f_out:
        for fp in fasta_path_array:
            with ngs.fasta.Fasta(fp, 'r') as f_in:
                for entry in f_in:
                    f_out.write(entry)
    return new_fasta_path


def split_and_index(
    fasta_path: str,
    index_prefix: str,
    n: int = 2,
    k: int = 31,
    temp_dir: str = 'tmp'
) -> Dict[str, str]:
    """Split a FASTA file into `n` parts and index each one.

    Args:
        fasta_path: Path to FASTA file
        index_prefix: Prefix of output kallisto indices
        n: Split the index into `n` files, defaults to `2`
        k: K-mer length, defaults to 31
        temp_dir: Path to temporary directory, defaults to `tmp`

    Returns:
        Dictionary containing path to generated index
    """
    fastas = []
    indices = []

    logger.info(f'Splitting {fasta_path} into {n} parts')
    size = int(os.path.getsize(fasta_path) / n) + 4

    with ngs.fasta.Fasta(fasta_path, 'r') as f_in:
        fasta_iter = iter(f_in)
        finished = False
        for i in range(n):
            fasta_part_path = get_temporary_filename(temp_dir)
            index_part_path = f'{index_prefix}.{i}'
            fastas.append(fasta_part_path)
            indices.append(index_part_path)

            with ngs.fasta.Fasta(fasta_part_path, 'w') as f_out:
                logger.debug(f'Writing {fasta_part_path}')
                while f_out.tell() < size:
                    try:
                        entry = next(fasta_iter)
                    except StopIteration:
                        finished = True
                        break
                    f_out.write(entry)

            if finished:
                break

    built = []
    for fasta_part_path, index_part_path in zip(fastas, indices):
        result = kallisto_index(
            fasta_part_path, index_part_path, k=k, temp_dir=temp_dir
        )
        built.append(result['index'])

    return {'indices': built}


@logger.namespaced('download')
def download_reference(
    species: str,
    workflow: str,
    files: Dict[str, str],
    temp_dir: str = 'tmp',
    overwrite: bool = False,
    k: int = 31
) -> Dict[str, str]:
    """Downloads a provided reference file from a static url.

    Args:
        species: Name of species
        workflow: Type of workflow (nac or standard)
        files: Dictionary that has the command-line option as keys and
            the path as values. used to determine if all the required
            paths to download the given reference have been provided
        temp_dir: Path to temporary directory, defaults to `tmp`
        overwrite: Overwrite an existing index file, defaults to `False`
        k: k-mer size, defaults to `31` (only `31` and `63` are supported)

    Returns:
        Dictionary containing paths to generated file(s)

    Raise:
        RefError: If the required options are not provided
    """
    results = {}
    species = species.lower()
    workflow = workflow.lower()
    if not ngs.utils.all_exists(*list(files.values())) or overwrite:
        # Make sure all the required file paths are there.
        if 'i' not in set(files.keys()) or 'g' not in set(files.keys()):
            raise RefError(
                'Following options are required to download reference: -i, -g'
            )
        if workflow == 'nac' and 'c1' not in set(files.keys()):
            raise RefError(
                'Following options are required to download nac reference: -c1'
            )
        if workflow == 'nac' and 'c2' not in set(files.keys()):
            raise RefError(
                'Following options are required to download nac reference: -c2'
            )
        if workflow != 'nac' and workflow != 'standard':
            raise RefError(
                f'The following workflow option is not supported: {workflow}'
            )

        long = ""
        if k == 63:
            long = "_long"
        elif k != 31:
            logger.info(
                "Only k-mer lengths 31 or 63 supported, defaulting to 31"
            )
        url = "https://github.com/pachterlab/kallisto-transcriptome-indices/"
        url = url + f'releases/download/v1/{species}_index_{workflow}{long}.tar.xz'
        path = os.path.join(temp_dir, os.path.basename(url))
        logger.info(
            'Downloading files for {} ({} workflow) from {} to {}'.format(
                species, workflow, url, path
            )
        )
        local_path = download_file(url, path)

        logger.info('Extracting files from {}'.format(local_path))
        with tarfile.open(local_path, 'r:xz') as f:

            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(
                tar, path=".", members=None, *, numeric_owner=False
            ):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(f, temp_dir)

        reference_files = {}
        reference_files.update({'i': "index.idx"})
        reference_files.update({'g': "t2g.txt"})
        if workflow == "nac":
            reference_files.update({'c1': "cdna.txt"})
            reference_files.update({'c2': "nascent.txt"})

        for option in reference_files:
            os.rename(
                os.path.join(temp_dir, reference_files[option]), files[option]
            )
            results.update({option: files[option]})
    else:
        logger.info(
            'Skipping download because some files already exist. Use the --overwrite flag to overwrite.'
        )
    return results


def decompress_file(path: str, temp_dir: str = 'tmp') -> str:
    """Decompress the given path if it is a .gz file. Otherwise, return the
    original path.

    Args:
        path: Path to the file

    Returns:
        Unaltered `path` if the file is not a .gz file, otherwise path to the
            uncompressed file
    """
    if path.endswith('.gz'):
        logger.info('Decompressing {} to {}'.format(path, temp_dir))
        return decompress_gzip(
            path,
            os.path.join(temp_dir,
                         os.path.splitext(os.path.basename(path))[0])
        )
    else:
        return path


def get_gtf_attribute_include_func(
    include: List[Dict[str, str]]
) -> Callable[[ngs.gtf.GtfEntry], bool]:
    """Helper function to create a filtering function to include certain GTF
    entries while processing. The returned function returns `True` if the
    entry should be included.

    Args:
        include: List of dictionaries representing key-value pairs of
            attributes to include

    Returns:
        Filter function
    """

    def include_func(entry):
        attributes = entry.attributes
        return any(
            all(attributes.get(key) == value
                for key, value in d.items())
            for d in include
        )

    return include_func


def get_gtf_attribute_exclude_func(
    exclude: List[Dict[str, str]]
) -> Callable[[ngs.gtf.GtfEntry], bool]:
    """Helper function to create a filtering function to exclude certain GTF
    entries while processing. The returned function returns `False` if the
    entry should be excluded.

    Args:
        exclude: List of dictionaries representing key-value pairs of
            attributes to exclude

    Returns:
        Filter function
    """

    def exclude_func(entry):
        attributes = entry.attributes
        return all(
            any(attributes.get(key) != value
                for key, value in d.items())
            for d in exclude
        )

    return exclude_func


@logger.namespaced('ref')
def ref(
    fasta_paths: Union[List[str], str],
    gtf_paths: Union[List[str], str],
    cdna_path: str,
    index_path: str,
    t2g_path: str,
    nucleus: bool = False,
    n: int = 1,
    k: Optional[int] = None,
    include: Optional[List[Dict[str, str]]] = None,
    exclude: Optional[List[Dict[str, str]]] = None,
    temp_dir: str = 'tmp',
    overwrite: bool = False,
    make_unique: bool = False,
    threads: int = 8,
    dlist: str = None,
    dlist_overhang: int = 1,
    aa: bool = False,
    max_ec_size: int = None,
) -> Dict[str, str]:
    """Generates files necessary to generate count matrices for single-cell RNA-seq.

    Args:
        fasta_paths: List of paths to genomic FASTA files
        gtf_paths: List of paths to GTF files
        cdna_path: Path to generate the cDNA FASTA file
        t2g_path: Path to output transcript-to-gene mapping
        nucleus: Whether to quantify single-nucleus RNA-seq, defaults to `False`
        n: Split the index into `n` files
        k: Override default kmer length 31, defaults to `None`
        include: List of dictionaries representing key-value pairs of
            attributes to include
        exclude: List of dictionaries representing key-value pairs of
            attributes to exclude
        temp_dir: Path to temporary directory, defaults to `tmp`
        overwrite: Overwrite an existing index file, defaults to `False`
        make_unique: Replace repeated target names with unique names, defaults to `False`
        threads: Number of threads to use, defaults to `8`
        dlist: Path to a FASTA-file containing sequences to mask from quantification,
            defaults to `None`
        dlist_overhang: The overhang to use for the D-list, defaults to `1`
        aa: Generate index from a FASTA-file containing amino acid sequences,
            defaults to `False`
        max_ec_size: Sets max size of equivalence class, defaults to `None`

    Returns:
        Dictionary containing paths to generated file(s)
    """
    dlist = get_dlist_fasta(dlist)
    if not isinstance(fasta_paths, list):
        fasta_paths = [fasta_paths]
    if not isinstance(gtf_paths, list):
        gtf_paths = [gtf_paths]
    include_func = get_gtf_attribute_include_func(
        include
    ) if include else lambda entry: True
    exclude_func = get_gtf_attribute_exclude_func(
        exclude
    ) if exclude else lambda entry: True
    filter_func = lambda entry: include_func(entry) and exclude_func(entry)

    results = {}
    cdnas = []
    target = "cDNA"
    if nucleus:
        target = "unprocessed transcript"

    if aa and not gtf_paths:
        logger.info(
            f'Skipping {target} FASTA generation because flag `--aa` was called without providing GTF file(s).'
        )

        if len(fasta_paths) > 1:
            raise RefError((
                'Option `--aa` does not support multiple FASTA files as input'
                'while no GTF file(s) provided'
            ))
        else:
            cdna_path = fasta_paths[0]

    elif (not ngs.utils.all_exists(cdna_path, t2g_path)) or overwrite:
        for fasta_path, gtf_path in zip(fasta_paths, gtf_paths):
            logger.info(f'Preparing {fasta_path}, {gtf_path}')
            # Parse GTF for gene and transcripts
            gene_infos, transcript_infos = ngs.gtf.genes_and_transcripts_from_gtf(
                gtf_path, use_version=True, filter_func=filter_func
            )

            # Split
            cdna_temp_path = get_temporary_filename(temp_dir)
            logger.info(
                f'Splitting genome {fasta_path} into {target} at {cdna_temp_path}'
            )
            if not nucleus:
                cdna_temp_path = ngs.fasta.split_genomic_fasta_to_cdna(
                    fasta_path, cdna_temp_path, gene_infos, transcript_infos
                )
            else:
                cdna_temp_path = ngs.fasta.split_genomic_fasta_to_nascent(
                    fasta_path, cdna_temp_path, gene_infos
                )
            cdnas.append(cdna_temp_path)

        logger.info(f'Concatenating {len(cdnas)} {target}s to {cdna_path}')
        cdna_path = concatenate_files(*cdnas, out_path=cdna_path)
        results.update({'cdna_fasta': cdna_path})

    else:
        logger.info(
            f'Skipping {target} FASTA generation because {cdna_path} already exists. Use --overwrite flag to overwrite'
        )

    if not glob.glob(f'{index_path}*') or overwrite:
        t2g_result = create_t2g_from_fasta(cdna_path, t2g_path, aa_flag=aa)
        results.update(t2g_result)
        if index_path.upper() == "NONE":
            return results

        if k and k != 31:
            logger.warning(
                f'Using provided k-mer length {k} instead of optimal length 31'
            )
        index_result = split_and_index(
            cdna_path, index_path, n=n, k=k or 31, temp_dir=temp_dir
        ) if n > 1 else kallisto_index(
            cdna_path,
            index_path,
            k=k or 31,
            threads=threads,
            dlist=dlist,
            dlist_overhang=dlist_overhang,
            aa=aa,
            make_unique=make_unique,
            max_ec_size=max_ec_size,
            temp_dir=temp_dir,
        )
        results.update(index_result)
    else:
        logger.info(
            'Skipping kallisto index because {} already exists. Use the --overwrite flag to overwrite.'
            .format(index_path)
        )

    return results


@logger.namespaced('ref_kite')
def ref_kite(
    feature_path: str,
    fasta_path: str,
    index_path: str,
    t2g_path: str,
    n: int = 1,
    k: Optional[int] = None,
    no_mismatches: bool = False,
    temp_dir: str = 'tmp',
    overwrite: bool = False,
    threads: int = 8
) -> Dict[str, str]:
    """Generates files necessary for feature barcoding with the KITE workflow.

    Args:
        feature_path: Path to TSV containing barcodes and feature names
        fasta_path: Path to generate fasta file containing all sequences
            that are 1 hamming distance from the provide barcodes (including
            the actual sequence)
        t2g_path: Path to output transcript-to-gene mapping
        n: Split the index into `n` files
        k: Override calculated optimal kmer length, defaults to `None`
        no_mismatches: Whether to generate hamming distance 1 variants,
            defaults to `False`
        temp_dir: Path to temporary directory, defaults to `tmp`
        overwrite: Overwrite an existing index file, defaults to `False`
        threads: Number of threads to use, defaults to `8`

    Returns:
        Dictionary containing paths to generated file(s)
    """
    results = {}
    feature_path = decompress_file(feature_path, temp_dir=temp_dir)
    logger.info('Generating mismatch FASTA at {}'.format(fasta_path))
    kite_path, length = generate_kite_fasta(
        feature_path, fasta_path, no_mismatches=no_mismatches
    )
    results.update({'fasta': kite_path})
    t2g_result = create_t2g_from_fasta(fasta_path, t2g_path)
    results.update(t2g_result)

    if not glob.glob(f'{index_path}*') or overwrite:
        optimal_k = length if length % 2 else length - 1
        if k and k != optimal_k:
            logger.warning(
                f'Using provided k-mer length {k} instead of calculated optimal length {optimal_k}'
            )
        index_result = split_and_index(
            kite_path, index_path, n=n, k=k or optimal_k, temp_dir=temp_dir
        ) if n > 1 else kallisto_index(
            kite_path,
            index_path,
            k=k or optimal_k,
            threads=threads,
            temp_dir=temp_dir
        )
        results.update(index_result)
    else:
        logger.info(
            'Skipping kallisto index because {} already exists. Use the --overwrite flag to overwrite.'
            .format(index_path)
        )
    return results


@logger.namespaced('ref_custom')
def ref_custom(
    fasta_paths: Union[List[str], str],
    index_path: str,
    k: Optional[int] = 31,
    threads: int = 8,
    dlist: str = None,
    dlist_overhang: int = 1,
    aa: bool = False,
    overwrite: bool = False,
    temp_dir: str = 'tmp',
    make_unique: bool = False,
    distinguish: bool = False,
) -> Dict[str, str]:
    """Generates files necessary for indexing custom targets.

    Args:
        fasta_paths: List of paths to FASTA files from which to extract k-mers
        index_path: Path to output kallisto index
        k: Override calculated optimal kmer length, defaults to `31`
        threads: Number of threads to use, defaults to `8`
        dlist: Path to a FASTA-file containing sequences to mask from quantification,
            defaults to `None`
        dlist_overhang: The overhang to use for the D-list, defaults to `1`
        aa: Generate index from a FASTA-file containing amino acid sequences,
            defaults to `False`
        overwrite: Overwrite an existing index file, defaults to `False`
        temp_dir: Path to temporary directory, defaults to `tmp`
        make_unique: Replace repeated target names with unique names, defaults to `False`
        skip_index: Skip index generation, defaults to `False`
        distinguish: Whether to index sequences by their shared name, defaults to `False`

    Returns:
        Dictionary containing paths to generated file(s)
    """
    dlist = get_dlist_fasta(dlist)
    if not isinstance(fasta_paths, list):
        fasta_paths = [fasta_paths]
    if k and k != 31:
        logger.warning(
            f'Using provided k-mer length {k} instead of optimal length 31'
        )
    else:
        k = 31

    results = {}

    if not glob.glob(f'{index_path}*') or overwrite:
        index_result = kallisto_index(
            ','.join(fasta_paths),
            index_path,
            k=k or 31,
            threads=threads,
            dlist=dlist,
            dlist_overhang=dlist_overhang,
            aa=aa,
            make_unique=make_unique,
            distinguish=distinguish,
            temp_dir=temp_dir
        )
        logger.info('Finished creating custom index')
        results.update(index_result)
    else:
        logger.info(
            'Skipping kallisto index because {} already exists. Use the --overwrite flag to overwrite.'
            .format(index_path)
        )

    return results


@logger.namespaced('ref_nac')
def ref_nac(
    fasta_paths: Union[List[str], str],
    gtf_paths: Union[List[str], str],
    cdna_path: str,
    intron_path: str,
    index_path: str,
    t2g_path: str,
    cdna_t2c_path: str,
    intron_t2c_path: str,
    nascent: bool = True,
    n: int = 1,
    k: Optional[int] = None,
    flank: Optional[int] = None,
    include: Optional[List[Dict[str, str]]] = None,
    exclude: Optional[List[Dict[str, str]]] = None,
    temp_dir: str = 'tmp',
    overwrite: bool = False,
    make_unique: bool = False,
    threads: int = 8,
    dlist: str = None,
    dlist_overhang: int = 1,
    max_ec_size: int = None
) -> Dict[str, str]:
    """Generates files necessary to generate RNA velocity matrices for single-cell RNA-seq.

    Args:
        fasta_paths: List of paths to genomic FASTA files
        gtf_paths: List of paths to GTF files
        cdna_path: Path to generate the cDNA FASTA file
        intron_path: Path to generate the intron or nascent FASTA file
        t2g_path: Path to output transcript-to-gene mapping
        cdna_t2c_path: Path to generate the cDNA transcripts-to-capture file
        intron_t2c_path: Path to generate the intron transcripts-to-capture file
        nascent: Obtain nascent/mature/ambiguous matrices, defaults to `True`
        n: Split the index into `n` files
        k: Override default kmer length (31), defaults to `None`
        flank: Number of bases to include from the flanking regions
            when generating the intron FASTA, defaults to `None`, which
            sets the flanking region to be k - 1 bases.
        include: List of dictionaries representing key-value pairs of
            attributes to include
        exclude: List of dictionaries representing key-value pairs of
            attributes to exclude
        temp_dir: Path to temporary directory, defaults to `tmp`
        overwrite: Overwrite an existing index file, defaults to `False`
        make_unique: Replace repeated target names with unique names, defaults to `False`
        threads: Number of threads to use, defaults to `8`
        dlist: Path to a FASTA-file containing sequences to mask from quantification,
            defaults to `None`
        dlist_overhang: The overhang to use for the D-list, defaults to `1`
        max_ec_size: Sets max size of equivalence class, defaults to `None`

    Returns:
        Dictionary containing paths to generated file(s)
    """
    dlist = get_dlist_fasta(dlist)
    if not isinstance(fasta_paths, list):
        fasta_paths = [fasta_paths]
    if not isinstance(gtf_paths, list):
        gtf_paths = [gtf_paths]
    include_func = get_gtf_attribute_include_func(
        include
    ) if include else lambda entry: True
    exclude_func = get_gtf_attribute_exclude_func(
        exclude
    ) if exclude else lambda entry: True
    filter_func = lambda entry: include_func(entry) and exclude_func(entry)

    results = {}
    cdnas = []
    introns = []
    cdna_t2cs = []
    intron_t2cs = []
    target = "intron"
    if nascent:
        target = "unprocessed transcript"
    if (not ngs.utils.all_exists(cdna_path, intron_path, t2g_path,
                                 cdna_t2c_path, intron_t2c_path)) or overwrite:
        for fasta_path, gtf_path in zip(fasta_paths, gtf_paths):
            logger.info(f'Preparing {fasta_path}, {gtf_path}')
            # Parse GTF for gene and transcripts
            gene_infos, transcript_infos = ngs.gtf.genes_and_transcripts_from_gtf(
                gtf_path, use_version=True, filter_func=filter_func
            )

            # Split cDNA
            cdna_temp_path = get_temporary_filename(temp_dir)
            logger.info(
                f'Splitting genome {fasta_path} into cDNA at {cdna_temp_path}'
            )
            cdna_temp_path = ngs.fasta.split_genomic_fasta_to_cdna(
                fasta_path, cdna_temp_path, gene_infos, transcript_infos
            )
            cdnas.append(cdna_temp_path)

            # cDNA t2c
            cdna_t2c_temp_path = get_temporary_filename(temp_dir)
            logger.info(
                f'Creating cDNA transcripts-to-capture at {cdna_t2c_temp_path}'
            )
            cdna_t2c_result = create_t2c(cdna_temp_path, cdna_t2c_temp_path)
            cdna_t2cs.append(cdna_t2c_result['t2c'])

            # Split intron
            intron_temp_path = get_temporary_filename(temp_dir)
            logger.info(
                f'Splitting genome into {target}s at {intron_temp_path}'
            )
            if not nascent:
                intron_temp_path = ngs.fasta.split_genomic_fasta_to_intron(
                    fasta_path,
                    intron_temp_path,
                    gene_infos,
                    transcript_infos,
                    flank=flank if flank is not None else k -
                    1 if k is not None else 30
                )
            else:
                intron_temp_path = ngs.fasta.split_genomic_fasta_to_nascent(
                    fasta_path, intron_temp_path, gene_infos
                )

            introns.append(intron_temp_path)

            # intron t2c
            intron_t2c_temp_path = get_temporary_filename(temp_dir)
            logger.info(
                f'Creating {target} transcripts-to-capture at {intron_t2c_temp_path}'
            )
            intron_t2c_result = create_t2c(
                intron_temp_path, intron_t2c_temp_path
            )
            intron_t2cs.append(intron_t2c_result['t2c'])

        # Concatenate
        logger.info(f'Concatenating {len(cdnas)} cDNA FASTAs to {cdna_path}')
        cdna_path = concatenate_files(*cdnas, out_path=cdna_path)
        logger.info(
            f'Concatenating {len(cdna_t2cs)} cDNA transcripts-to-captures to {cdna_t2c_path}'
        )
        cdna_t2c_path = concatenate_files(*cdna_t2cs, out_path=cdna_t2c_path)
        logger.info(
            f'Concatenating {len(introns)} {target} FASTAs to {intron_path}'
        )
        intron_path = concatenate_files(*introns, out_path=intron_path)
        logger.info(
            f'Concatenating {len(intron_t2cs)} {target} transcripts-to-captures to {intron_t2c_path}'
        )
        intron_t2c_path = concatenate_files(
            *intron_t2cs, out_path=intron_t2c_path
        )
        results.update({
            'cdna_fasta': cdna_path,
            'cdna_t2c': cdna_t2c_path,
            'intron_fasta': intron_path,
            'intron_t2c': intron_t2c_path
        })

    else:
        logger.info(
            'Skipping cDNA and {target} FASTA generation because files already exist. Use --overwrite flag to overwrite'
        )

    if not glob.glob(f'{index_path}*') or overwrite:
        # Concatenate cDNA and intron fastas to generate T2G and build index
        combined_path = get_temporary_filename(temp_dir)
        logger.info(
            f'Concatenating cDNA and {target} FASTAs to {combined_path}'
        )
        combined_path = concatenate_files(
            cdna_path, intron_path, out_path=combined_path
        )
        t2g_result = create_t2g_from_fasta(combined_path, t2g_path)
        results.update(t2g_result)
        if index_path.upper() == "NONE":
            return results

        if k and k != 31:
            logger.warning(
                f'Using provided k-mer length {k} instead of optimal length 31'
            )

        # If n = 1, make single index
        # if n = 2, make two indices, one for spliced and another for unspliced
        # if n > 2, make n indices, one for spliced, another n - 1 for unspliced
        # if nascent, make single index (nascent/mature/ambiguous)
        if nascent:
            index_result = kallisto_index(
                combined_path,
                index_path,
                k=k or 31,
                threads=threads,
                dlist=dlist,
                dlist_overhang=dlist_overhang,
                make_unique=make_unique,
                max_ec_size=max_ec_size,
                temp_dir=temp_dir
            )
        elif n == 1:
            index_result = kallisto_index(
                combined_path,
                index_path,
                k=k or 31,
                threads=threads,
                dlist=dlist,
                dlist_overhang=dlist_overhang,
                make_unique=make_unique,
                max_ec_size=max_ec_size,
                temp_dir=temp_dir
            )
        else:
            cdna_index_result = kallisto_index(
                cdna_path,
                f'{index_path}_cdna',
                k=k or 31,
                threads=threads,
                dlist=dlist,
                dlist_overhang=dlist_overhang,
                make_unique=make_unique,
                max_ec_size=max_ec_size,
                temp_dir=temp_dir
            )
            if n == 2:
                intron_index_result = kallisto_index(
                    intron_path,
                    f'{index_path}_intron',
                    k=k or 31,
                    threads=threads,
                    dlist=dlist,
                    dlist_overhang=dlist_overhang,
                    make_unique=make_unique,
                    max_ec_size=max_ec_size,
                    temp_dir=temp_dir
                )
                index_result = {
                    'indices': [
                        cdna_index_result['index'], intron_index_result['index']
                    ]
                }
            else:
                split_index_result = split_and_index(
                    intron_path,
                    f'{index_path}_intron',
                    n=n - 1,
                    k=k or 31,
                    temp_dir=temp_dir
                )
                index_result = {
                    'indices': [
                        cdna_index_result['index'],
                        *split_index_result['indices']
                    ]
                }
        results.update(index_result)
    else:
        logger.info(
            'Skipping kallisto index because {} already exists. Use the --overwrite flag to overwrite.'
            .format(index_path)
        )

    return results


@logger.namespaced('ref_lamanno')
def ref_lamanno(
    fasta_paths: Union[List[str], str],
    gtf_paths: Union[List[str], str],
    cdna_path: str,
    intron_path: str,
    index_path: str,
    t2g_path: str,
    cdna_t2c_path: str,
    intron_t2c_path: str,
    n: int = 1,
    k: Optional[int] = None,
    flank: Optional[int] = None,
    include: Optional[List[Dict[str, str]]] = None,
    exclude: Optional[List[Dict[str, str]]] = None,
    temp_dir: str = 'tmp',
    overwrite: bool = False,
    threads: int = 8,
) -> Dict[str, str]:
    """RNA velocity index (DEPRECATED).

    Args:
        fasta_paths: List of paths to genomic FASTA files
        gtf_paths: List of paths to GTF files
        cdna_path: Path to generate the cDNA FASTA file
        intron_path: Path to generate the intron FASTA file
        t2g_path: Path to output transcript-to-gene mapping
        cdna_t2c_path: Path to generate the cDNA transcripts-to-capture file
        intron_t2c_path: Path to generate the intron transcripts-to-capture file
        n: Split the index into `n` files
        k: Override default kmer length (31), defaults to `None`
        flank: Number of bases to include from the flanking regions
            when generating the intron FASTA, defaults to `None`, which
            sets the flanking region to be k - 1 bases.
        include: List of dictionaries representing key-value pairs of
            attributes to include
        exclude: List of dictionaries representing key-value pairs of
            attributes to exclude
        temp_dir: Path to temporary directory, defaults to `tmp`
        overwrite: Overwrite an existing index file, defaults to `False`
        threads: Number of threads to use, defaults to `8`

    Returns:
        Dictionary containing paths to generated file(s)
    """
    if not isinstance(fasta_paths, list):
        fasta_paths = [fasta_paths]
    if not isinstance(gtf_paths, list):
        gtf_paths = [gtf_paths]
    include_func = get_gtf_attribute_include_func(
        include
    ) if include else lambda entry: True
    exclude_func = get_gtf_attribute_exclude_func(
        exclude
    ) if exclude else lambda entry: True
    filter_func = lambda entry: include_func(entry) and exclude_func(entry)

    results = {}
    cdnas = []
    introns = []
    cdna_t2cs = []
    intron_t2cs = []
    if (not ngs.utils.all_exists(cdna_path, intron_path, t2g_path,
                                 cdna_t2c_path, intron_t2c_path)) or overwrite:
        for fasta_path, gtf_path in zip(fasta_paths, gtf_paths):
            logger.info(f'Preparing {fasta_path}, {gtf_path}')
            # Parse GTF for gene and transcripts
            gene_infos, transcript_infos = ngs.gtf.genes_and_transcripts_from_gtf(
                gtf_path, use_version=True, filter_func=filter_func
            )

            # Split cDNA
            cdna_temp_path = get_temporary_filename(temp_dir)
            logger.info(
                f'Splitting genome {fasta_path} into cDNA at {cdna_temp_path}'
            )
            cdna_temp_path = ngs.fasta.split_genomic_fasta_to_cdna(
                fasta_path, cdna_temp_path, gene_infos, transcript_infos
            )
            cdnas.append(cdna_temp_path)

            # cDNA t2c
            cdna_t2c_temp_path = get_temporary_filename(temp_dir)
            logger.info(
                f'Creating cDNA transcripts-to-capture at {cdna_t2c_temp_path}'
            )
            cdna_t2c_result = create_t2c(cdna_temp_path, cdna_t2c_temp_path)
            cdna_t2cs.append(cdna_t2c_result['t2c'])

            # Split intron
            intron_temp_path = get_temporary_filename(temp_dir)
            logger.info(f'Splitting genome into introns at {intron_temp_path}')
            intron_temp_path = ngs.fasta.split_genomic_fasta_to_intron(
                fasta_path,
                intron_temp_path,
                gene_infos,
                transcript_infos,
                flank=flank if flank is not None else k -
                1 if k is not None else 30
            )
            introns.append(intron_temp_path)

            # intron t2c
            intron_t2c_temp_path = get_temporary_filename(temp_dir)
            logger.info(
                f'Creating intron transcripts-to-capture at {intron_t2c_temp_path}'
            )
            intron_t2c_result = create_t2c(
                intron_temp_path, intron_t2c_temp_path
            )
            intron_t2cs.append(intron_t2c_result['t2c'])

        # Concatenate
        logger.info(f'Concatenating {len(cdnas)} cDNA FASTAs to {cdna_path}')
        cdna_path = concatenate_files(*cdnas, out_path=cdna_path)
        logger.info(
            f'Concatenating {len(cdna_t2cs)} cDNA transcripts-to-captures to {cdna_t2c_path}'
        )
        cdna_t2c_path = concatenate_files(*cdna_t2cs, out_path=cdna_t2c_path)
        logger.info(
            f'Concatenating {len(introns)} intron FASTAs to {intron_path}'
        )
        intron_path = concatenate_files(*introns, out_path=intron_path)
        logger.info(
            f'Concatenating {len(intron_t2cs)} intron transcripts-to-captures to {intron_t2c_path}'
        )
        intron_t2c_path = concatenate_files(
            *intron_t2cs, out_path=intron_t2c_path
        )
        results.update({
            'cdna_fasta': cdna_path,
            'cdna_t2c': cdna_t2c_path,
            'intron_fasta': intron_path,
            'intron_t2c': intron_t2c_path
        })

    else:
        logger.info(
            'Skipping cDNA and intron FASTA generation because files already exist. Use --overwrite flag to overwrite'
        )

    if not glob.glob(f'{index_path}*') or overwrite:
        # Concatenate cDNA and intron fastas to generate T2G and build index
        combined_path = get_temporary_filename(temp_dir)
        logger.info(f'Concatenating cDNA and intron FASTAs to {combined_path}')
        combined_path = concatenate_files(
            cdna_path, intron_path, out_path=combined_path
        )
        t2g_result = create_t2g_from_fasta(combined_path, t2g_path)
        results.update(t2g_result)

        if k and k != 31:
            logger.warning(
                f'Using provided k-mer length {k} instead of optimal length 31'
            )

        # If n = 1, make single index
        # if n = 2, make two indices, one for spliced and another for unspliced
        # if n > 2, make n indices, one for spliced, another n - 1 for unspliced
        if n == 1:
            index_result = kallisto_index(
                combined_path,
                index_path,
                k=k or 31,
                threads=threads,
                temp_dir=temp_dir
            )
        else:
            cdna_index_result = kallisto_index(
                cdna_path, f'{index_path}_cdna', k=k or 31, temp_dir=temp_dir
            )
            if n == 2:
                intron_index_result = kallisto_index(
                    intron_path,
                    f'{index_path}_intron',
                    k=k or 31,
                    temp_dir=temp_dir
                )
                index_result = {
                    'indices': [
                        cdna_index_result['index'], intron_index_result['index']
                    ]
                }
            else:
                split_index_result = split_and_index(
                    intron_path,
                    f'{index_path}_intron',
                    n=n - 1,
                    k=k or 31,
                    temp_dir=temp_dir
                )
                index_result = {
                    'indices': [
                        cdna_index_result['index'],
                        *split_index_result['indices']
                    ]
                }
        results.update(index_result)
    else:
        logger.info(
            'Skipping kallisto index because {} already exists. Use the --overwrite flag to overwrite.'
            .format(index_path)
        )

    return results
