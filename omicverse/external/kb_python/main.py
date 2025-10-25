import argparse
import logging
import os
import shutil
import sys
import textwrap
import warnings
from typing import Tuple

from . import __version__
from .config import (
    COMPILED_DIR,
    get_bustools_binary_path,
    get_kallisto_binary_path,
    is_dry,
    no_validate,
    PACKAGE_PATH,
    set_dry,
    set_bustools_binary_path,
    set_kallisto_binary_path,
    set_special_kallisto_binary,
    get_provided_kallisto_path,
    TECHNOLOGIES,
    TEMP_DIR,
    UnsupportedOSError,
)
from .compile import compile
from .constants import INFO_FILENAME
from .logging import logger
from .ref import download_reference, ref, ref_kite, ref_lamanno, ref_nac, ref_custom
from .utils import (
    get_bustools_version,
    get_kallisto_version,
    make_directory,
    open_as_text,
    remove_directory,
    whitelist_provided,
)


def test_binaries() -> Tuple[bool, bool]:
    """Test whether kallisto and bustools binaries are executable.

    Internally, this function calls :func:`utils.get_kallisto_version` and
    :func:`utils.get_bustools_version`, both of which return `None` if there is
    something wrong with their respective binaries.

    Returns:
        A tuple of two booleans indicating kallisto and bustools binaries.
    """
    kallisto_ok = True
    try:
        kallisto_ok = get_kallisto_version() is not None
    except Exception:
        kallisto_ok = False
    bustools_ok = True
    try:
        bustools_ok = get_bustools_version() is not None
    except Exception:
        bustools_ok = False

    return kallisto_ok, bustools_ok


def get_binary_info() -> str:
    """Get information on the binaries that will be used for commands.

    Returns:
        `kallisto` and `bustools` binary versions and paths.
    """
    kallisto_version = '.'.join(str(i) for i in get_kallisto_version())
    bustools_version = '.'.join(str(i) for i in get_bustools_version())
    return (
        f'kallisto: {kallisto_version} ({get_kallisto_binary_path()})\n'
        f'bustools: {bustools_version} ({get_bustools_binary_path()})'
    )


def display_info():
    """Displays kb, kallisto and bustools version + citation information, along
    with a brief description and examples.
    """
    info = f'kb_python {__version__}\n{get_binary_info()}'
    with open(os.path.join(PACKAGE_PATH, INFO_FILENAME), 'r') as f:
        print(
            '{}\n{}'.format(
                info, '\n'.join([
                    line.strip()
                    if line.startswith('(') else textwrap.fill(line, width=80)
                    for line in f.readlines()
                ])
            )
        )
    sys.exit(1)


def display_technologies():
    """Displays a list of supported technologies along with whether kb provides
    a whitelist for that technology and the FASTQ argument order for kb count.
    """
    headers = ['name', 'description', 'on-list', 'barcode', 'umi', 'cDNA']
    rows = [headers]

    print('List of supported single-cell technologies\n')
    print('Positions syntax: `input file index, start position, end position`')
    print('When start & end positions are None, refers to the entire file')
    print(
        'Custom technologies may be defined by providing a kallisto-supported '
        'technology string\n(see https://pachterlab.github.io/kallisto/manual)\n'
    )
    for t in TECHNOLOGIES:
        if not t.show:
            continue
        chem = t.chemistry
        row = [
            t.name,
            t.description,
            'yes' if chem.has_whitelist else '',
            ' '.join(str(_def) for _def in chem.barcode_parser)
            if chem.has_barcode else '',
            ' '.join(str(_def)
                     for _def in chem.umi_parser) if chem.has_umi else '',
            ' '.join(str(_def) for _def in chem.cdna_parser),
        ]
        rows.append(row)

    max_lens = []
    for i in range(len(headers)):
        max_lens.append(len(headers[i]))
        for row in rows[1:]:
            max_lens[i] = max(max_lens[i], len(row[i]))

    rows.insert(1, ['-' * l for l in max_lens])  # noqa
    for row in rows:
        for col, l in zip(row, max_lens):
            print(col.ljust(l + 4), end='')
        print()
    sys.exit(1)


def parse_compile(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    temp_dir: str = 'tmp'
):
    """Parser for the `compile` command.

    Args:
        parser: The argument parser
        args: Parsed command-line arguments
    """
    # target must be all when --view is used
    if args.view and args.target is not None:
        parser.error(
            '`target` must not be be provided when `--view` is provided.'
        )

    # target must not be all when --url is provided
    if args.url and args.target == 'all':
        parser.error('`target` must not be `all` when `--url` is provided.')

    # --view or --remove may not be specified with -o
    if args.o and (args.view or args.remove):
        parser.error('`-o` may not be used with `--view` or `--remove`')
    if args.cmake_arguments and (args.view or args.remove):
        parser.error(
            '`--cmake-arguments` may not be used with `--view` or `--remove`'
        )

    if args.remove:
        if args.target in ('kallisto', 'all'):
            shutil.rmtree(
                os.path.join(COMPILED_DIR, 'kallisto'), ignore_errors=True
            )
        if args.target in ('bustools', 'all'):
            shutil.rmtree(
                os.path.join(COMPILED_DIR, 'bustools'), ignore_errors=True
            )
    elif args.view:
        print(get_binary_info())
        sys.exit(1)
    else:
        if args.target not in ('kallisto', 'bustools', 'all'):
            parser.error(
                '`target` must be one of `kallisto`, `bustools`, `all`'
            )

        compile(
            args.target,
            out_dir=args.o,
            cmake_arguments=args.cmake_arguments,
            url=args.url,
            ref=args.ref,
            overwrite=args.overwrite,
            temp_dir=temp_dir
        )


def parse_ref(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    temp_dir: str = 'tmp'
):
    """Parser for the `ref` command.

    Args:
        parser: The argument parser
        args: Parsed command-line arguments
    """
    dlist = None
    aa = False
    if args.k is not None:
        if args.k < 0 or not args.k % 2:
            parser.error('K-mer length must be a positive odd integer.')
    if args.d_list is None:
        if args.aa or args.workflow == 'custom':
            dlist = None
        else:
            # Use whole genome for dlist
            dlist = str(args.fasta)
    elif args.d_list.upper() != 'NONE':
        dlist = args.d_list
    if args.aa:
        aa = args.aa
    if args.fasta:
        args.fasta = args.fasta.split(',')
    if args.gtf:
        args.gtf = args.gtf.split(',')
    if not args.gtf and (args.aa or args.workflow == 'custom'):
        args.gtf = []
    if (args.fasta and args.gtf) and len(args.fasta) != len(args.gtf):
        if args.workflow != 'custom':
            parser.error(
                'There must be the same number of FASTAs as there are GTFs.'
            )

    # Parse include/exclude KEY:VALUE pairs
    include = []
    exclude = []
    if args.include_attribute:
        for kv in args.include_attribute:
            key, value = kv.split(':')
            if kv.count(':') != 1 or not key or not value:
                parser.error(f'Malformed KEY:VALUE pair `{kv}`')
            include.append({key: value})
    if args.exclude_attribute:
        for kv in args.exclude_attribute:
            key, value = kv.split(':')
            if kv.count(':') != 1 or not key or not value:
                parser.error(f'Malformed KEY:VALUE pair `{kv}`')
            exclude.append({key: value})

    if args.d is not None:
        # Options that are files.
        options = ['i', 'g', 'c1', 'c2']
        files = {
            option: getattr(args, option)
            for option in options
            if getattr(args, option) is not None
        }
        download_reference(
            args.d,
            args.workflow,
            files,
            overwrite=args.overwrite,
            temp_dir=temp_dir,
            k=31 if not args.k else args.k
        )
    elif args.workflow == 'nac':
        ref_nac(
            args.fasta,
            args.gtf,
            args.f1,
            args.f2,
            args.i,
            args.g,
            args.c1,
            args.c2,
            k=args.k,
            flank=args.flank,
            include=include,
            exclude=exclude,
            threads=args.t,
            dlist=dlist,
            dlist_overhang=args.d_list_overhang,
            overwrite=args.overwrite,
            make_unique=args.make_unique,
            temp_dir=temp_dir,
            max_ec_size=args.ec_max_size
        )
    elif args.workflow in {'lamanno', 'nucleus'}:
        if args.d_list is not None:
            parser.error("d-list incompatible with lamanno/nucleus")
        ref_lamanno(
            args.fasta,
            args.gtf,
            args.f1,
            args.f2,
            args.i,
            args.g,
            args.c1,
            args.c2,
            k=args.k,
            flank=args.flank,
            include=include,
            exclude=exclude,
            overwrite=args.overwrite,
            temp_dir=temp_dir,
            threads=args.t
        )
    else:
        # Report extraneous options
        velocity_only = ['f2', 'c1', 'c2', 'flank']
        for arg in velocity_only:
            if getattr(args, arg):
                parser.error(
                    f'Option `{arg}` is not supported for workflow `{args.workflow}`'
                )

        if args.workflow == 'kite':
            if args.include_attribute or args.exclude_attribute:
                parser.error(
                    '`--include-attribute` or `--exclude-attribute` may not be used '
                    f'for workflow `{args.workflow}`'
                )
            if args.d_list:
                parser.error(
                    f'`--d-list` may not be used for workflow `{args.workflow}`'
                )

            ref_kite(
                args.feature,
                args.f1,
                args.i,
                args.g,
                k=args.k,
                no_mismatches=args.no_mismatches,
                threads=args.t,
                overwrite=args.overwrite,
                temp_dir=temp_dir
            )
        elif args.workflow == 'custom':
            if aa and args.distinguish:
                parser.error('`--aa` may not be used with --distinguish')
            ref_custom(
                args.fasta,
                args.i,
                k=args.k,
                threads=args.t,
                dlist=dlist,
                dlist_overhang=args.d_list_overhang,
                aa=aa,
                overwrite=args.overwrite,
                temp_dir=temp_dir,
                make_unique=args.make_unique,
                distinguish=args.distinguish
            )
        else:
            ref(
                args.fasta,
                args.gtf,
                args.f1,
                args.i,
                args.g,
                nucleus=False,
                k=args.k,
                include=include,
                exclude=exclude,
                threads=args.t,
                dlist=dlist,
                dlist_overhang=args.d_list_overhang,
                aa=aa,
                overwrite=args.overwrite,
                make_unique=args.make_unique,
                temp_dir=temp_dir,
                max_ec_size=args.ec_max_size
            )


def parse_count(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    temp_dir: str = 'tmp'
):
    """Parser for the `count` command.

    Args:
        parser: The argument parser
        args: Parsed command-line arguments
    """
    if args.report:
        logger.warning((
            'Using `--report` may cause `kb` to exceed maximum memory specified '
            'and crash for large count matrices.'
        ))

    if args.filter_threshold and args.filter != 'bustools':
        parser.error(
            'Option `--filter-threshold` may only be used with `--filter bustools`.'
        )

    if args.tcc and args.cellranger:
        parser.error(
            'TCC matrices can not be converted to cellranger-compatible format.'
        )
    if args.tcc and args.report:
        logger.warning(
            'Plots for TCC matrices have not yet been implemented. '
            'The HTML report will not contain any plots.'
        )
    # Note: We are currently not supporting --genomebam
    if args.genomebam:
        parser.error('--genomebam is not currently supported')
    if args.genomebam and not args.gtf:
        parser.error('`--gtf` must be provided when using `--genomebam`.')
    if args.genomebam and not args.chromosomes:
        logger.warning(
            '`--chromosomes` is recommended when using `--genomebam`'
        )

    # Check quant-tcc options
    if args.matrix_to_files and args.matrix_to_directories:
        parser.error(
            '`--matrix-to-files` cannot be used with `--matrix-to-directories`.'
        )

    # Check if batch TSV was provided.
    batch_path = None
    if len(args.fastqs) == 1:
        try:
            with open_as_text(args.fastqs[0], 'r') as f:
                if not f.readline().startswith('@'):
                    batch_path = args.fastqs[0]
        except Exception:
            pass

    if args.inleaved:
        batch_path = None

    args.x = args.x.strip()

    if '%' in args.x:
        x_split = args.x.split('%')
        args.x = x_split[0]
        if args.strand is None:
            if x_split[1].upper() == "UNSTRANDED":
                args.strand = "unstranded"
            elif x_split[1].upper() == "FORWARD":
                args.strand = "forward"
            elif x_split[1].upper() == "REVERSE":
                args.strand = "reverse"
        if args.parity is None and len(x_split) > 2:
            if x_split[2].upper() == 'PAIRED':
                args.parity = "paired"
            else:
                args.parity = "single"

    demultiplexed = False
    if args.x.upper() == 'DEFAULT' or args.x.upper() == 'BULK':
        args.x = 'BULK'
        demultiplexed = True
    if args.x[0] == '-':
        # Custom technology where no barcodes exist
        demultiplexed = True

    if args.batch_barcodes and batch_path is None:
        parser.error(
            '`--batch-barcodes` can only be used if batch file supplied'
        )
    if args.batch_barcodes and demultiplexed:
        if args.x.upper() == 'DEFAULT' or args.x.upper() == 'BULK':
            parser.error(
                f'`--batch-barcodes` may not be used for technology {args.x}'
            )
    if args.batch_barcodes and args.w is None and not whitelist_provided(
            args.x.upper()) and not demultiplexed:
        parser.error(
            f'`--batch-barcodes` may not be used for technology {args.x} without on-list'
        )
    if args.batch_barcodes and args.filter:
        parser.error('`--batch-barcodes` may not be used with --filter')
    if args.x.upper() in ('BULK', 'SMARTSEQ2', 'SMARTSEQ3') and args.em:
        parser.error(f'`--em` may not be used for technology {args.x}')
    if args.x.upper() in ('BULK', 'SMARTSEQ2'):
        # Check unsupported options
        unsupported = ['filter']
        for arg in unsupported:
            if getattr(args, arg):
                parser.error(
                    f'Argument `{arg}` is not supported for technology `{args.x}`.'
                )

        if not args.parity:
            parser.error(
                f'`--parity` must be provided for technology `{args.x}`.'
            )

        if not batch_path and not demultiplexed:
            logger.warning(
                f'FASTQs were provided for technology `{args.x}`. '
                'Assuming multiplexed samples. For demultiplexed samples, provide '
                'a batch textfile or specify `bulk` as the technology.'
            )
        elif batch_path:
            # If `single`, then each row must contain 2 columns. If `paired`,
            # each row must contain 3 columns.
            target = 2 + (args.parity == 'paired')
            with open(batch_path, 'r') as f:
                for i, line in enumerate(f):
                    if line.isspace() or line.startswith('#'):
                        continue
                    sep = '\t' if '\t' in line else ' '
                    columns = len(line.split(sep))
                    if target != columns:
                        parser.error(
                            f'Batch file {batch_path} line {i} contains wrong '
                            f'number of columns. Expected {target} for '
                            f'`--parity {args.parity} but got {columns}.'
                        )

        if args.parity == 'single':
            if args.tcc:
                if (args.fragment_l is None) ^ (args.fragment_s is None):
                    parser.error(
                        'Both or neither `--fragment-l` and `--fragment-s` must be '
                        'provided for single-end reads with TCC output.'
                    )

                if args.fragment_l is None and args.fragment_s is None:
                    logger.warning(
                        '`--fragment-l` and `--fragment-s` not provided. '
                        'Assuming all transcripts have the exact same length.'
                    )
            elif (args.fragment_l is not None) or (args.fragment_s is not None):
                parser.error(
                    '`--fragment-l` and `--fragment-s` may only be used with `--tcc`.'
                )

        elif args.parity == 'paired':
            if args.fragment_l is not None or args.fragment_s is not None:
                parser.error(
                    '`--fragment-l` or `--fragment-s` may not be provided for '
                    'paired-end reads.'
                )
    elif args.x.upper() == 'SMARTSEQ3':
        unsupported = [
            'filter', 'parity', 'fragment-l', 'fragment-s', 'report',
            'cellranger'
        ]
        for arg in unsupported:
            if getattr(args, arg.replace('-', '_')):
                parser.error(
                    f'Argument `{arg}` is not supported for technology `{args.x}`.'
                )

            # Batch file not supported
            if batch_path:
                parser.error(
                    f'Technology {args.x} does not support a batch file.'
                )
    else:
        # Check unsupported options
        unsupported = ['fragment-l', 'fragment-s']
        for arg in unsupported:
            if getattr(args, arg.replace('-', '_')):
                parser.error(
                    f'Argument `{arg}` is not supported for technology `{args.x}`.'
                )

        if args.fragment_l is not None or args.fragment_s is not None:
            parser.error(
                '`--fragment-l` and `--fragment-s` may only be provided with '
                '`BULK` and `SMARTSEQ2` technologies.'
            )

    from .constants import VELOCYTO_LOOM_NAMES
    loom_names = args.loom_names
    if args.loom_names.upper().strip() == 'VELOCYTO':
        loom_names = VELOCYTO_LOOM_NAMES
    loom_names = [x.strip() for x in loom_names.split(',')]
    if '' in loom_names or len(loom_names) != 2:
        parser.error('`--loom-names` is invalid')

    if args.workflow == 'nac':
        # Smartseq can not be used with nac.
        if args.x.upper() in ('SMARTSEQ',):
            parser.error(
                f'Technology `{args.x}` can not be used with workflow {args.workflow}.'
            )
        if args.aa:
            parser.error(
                f'Option `--aa` cannot be used with workflow {args.workflow}.'
            )
        from .count import count_nac
        count_nac(
            args.i,
            args.g,
            args.c1,
            args.c2,
            args.x,
            args.o,
            batch_path or args.fastqs,
            args.w,
            args.r,
            tcc=args.tcc,
            mm=args.mm,
            filter=args.filter,
            filter_threshold=args.filter_threshold,
            threads=args.t,
            memory=args.m,
            overwrite=args.overwrite,
            loom=args.loom,
            loom_names=loom_names,
            h5ad=args.h5ad,
            cellranger=args.cellranger,
            report=args.report,
            inspect=not args.no_inspect,
            temp_dir=temp_dir,
            fragment_l=args.fragment_l,
            fragment_s=args.fragment_s,
            paired=args.parity == 'paired',
            genomebam=args.genomebam,
            strand=args.strand,
            umi_gene=args.x.upper() not in ('BULK', 'SMARTSEQ2'),
            em=args.em,
            by_name=args.gene_names,
            sum_matrices=args.sum,
            gtf_path=args.gtf,
            chromosomes_path=args.chromosomes,
            inleaved=args.inleaved,
            demultiplexed=demultiplexed,
            batch_barcodes=args.batch_barcodes,
            numreads=args.N,
            store_num=args.num,
            lr=args.long,
            lr_thresh=args.threshold,
            lr_error_rate=args.error_rate,
            lr_platform=args.platform,
            union=args.union,
            no_jump=args.no_jump,
            quant_umis=args.quant_umis,
            keep_flags=args.keep_flags,
            exact_barcodes=args.exact_barcodes
        )
    elif args.workflow in {'nucleus', 'lamanno'}:
        # Smartseq can not be used with lamanno or nucleus.
        if args.x.upper() in ('SMARTSEQ',):
            parser.error(
                f'Technology `{args.x}` can not be used with workflow {args.workflow}.'
            )
        if args.sum != "none":
            parser.error('--sum incompatible with lamanno/nucleus')
        if args.x.upper() == 'SMARTSEQ3':
            from .count import count_velocity_smartseq3
            count_velocity_smartseq3(
                args.i,
                args.g,
                args.c1,
                args.c2,
                args.o,
                args.fastqs,
                args.w,
                tcc=args.tcc,
                mm=args.mm,
                temp_dir=temp_dir,
                threads=args.t,
                memory=args.m,
                overwrite=args.overwrite,
                loom=args.loom,
                h5ad=args.h5ad,
                inspect=not args.no_inspect,
                strand=args.strand,
                by_name=args.gene_names
            )
        else:
            from .count import count_velocity
            count_velocity(
                args.i,
                args.g,
                args.c1,
                args.c2,
                args.x,
                args.o,
                batch_path or args.fastqs,
                args.w,
                tcc=args.tcc,
                mm=args.mm,
                filter=args.filter,
                filter_threshold=args.filter_threshold,
                threads=args.t,
                memory=args.m,
                overwrite=args.overwrite,
                loom=args.loom,
                h5ad=args.h5ad,
                cellranger=args.cellranger,
                report=args.report,
                inspect=not args.no_inspect,
                nucleus=args.workflow == 'nucleus',
                temp_dir=temp_dir,
                fragment_l=args.fragment_l,
                fragment_s=args.fragment_s,
                paired=args.parity == 'paired',
                strand=args.strand,
                umi_gene=args.x.upper() not in ('BULK', 'SMARTSEQ2'),
                em=args.em,
                by_name=args.gene_names
            )
    else:
        if args.workflow == 'kite:10xFB' and args.x.upper() != '10XV3':
            parser.error(
                '`kite:10xFB` workflow is only supported with technology `10XV3`'
            )

        from .count import count
        count(
            args.i,
            args.g,
            args.x,
            args.o,
            batch_path or args.fastqs,
            args.w,
            args.r,
            tcc=args.tcc,
            mm=args.mm,
            filter=args.filter,
            filter_threshold=args.filter_threshold,
            kite='kite' in args.workflow,
            FB='10xFB' in args.workflow,
            threads=args.t,
            memory=args.m,
            overwrite=args.overwrite,
            loom=args.loom,
            loom_names=loom_names,
            h5ad=args.h5ad,
            cellranger=args.cellranger,
            report=args.report,
            inspect=not args.no_inspect,
            temp_dir=temp_dir,
            fragment_l=args.fragment_l,
            fragment_s=args.fragment_s,
            paired=args.parity == 'paired',
            genomebam=args.genomebam,
            aa=args.aa,
            strand=args.strand,
            umi_gene=args.x.upper() not in ('BULK', 'SMARTSEQ2'),
            em=args.em,
            by_name=args.gene_names,
            gtf_path=args.gtf,
            chromosomes_path=args.chromosomes,
            inleaved=args.inleaved,
            demultiplexed=demultiplexed,
            batch_barcodes=args.batch_barcodes,
            bootstraps=args.bootstraps,
            matrix_to_files=args.matrix_to_files,
            matrix_to_directories=args.matrix_to_directories,
            no_fragment=args.no_fragment,
            numreads=args.N,
            store_num=args.num,
            lr=args.long,
            lr_thresh=args.threshold,
            lr_error_rate=args.error_rate,
            lr_platform=args.platform,
            union=args.union,
            no_jump=args.no_jump,
            quant_umis=args.quant_umis,
            keep_flags=args.keep_flags,
            exact_barcodes=args.exact_barcodes
        )


def parse_extract(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    temp_dir: str = 'tmp'
):
    """Parser for the `extract` command.

    Args:
        parser: The argument parser
        args: Parsed command-line arguments
    """

    from .extract import extract
    extract(
        fastq=args.fastq,
        index_path=args.i,
        targets=args.targets,
        target_type=args.target_type,
        extract_all=args.extract_all,
        extract_all_fast=args.extract_all_fast,
        extract_all_unmapped=args.extract_all_unmapped,
        out_dir=args.o,
        mm=args.mm,
        t2g_path=args.g,
        temp_dir=temp_dir,
        threads=args.t,
        aa=args.aa,
        strand=args.strand,
        numreads=args.N
    )


COMMAND_TO_FUNCTION = {
    'compile': parse_compile,
    'ref': parse_ref,
    'count': parse_count,
    'extract': parse_extract,
}


def setup_info_args(
    parser: argparse.ArgumentParser, parent: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Helper function to set up a subparser for the `info` command.

    Args:
        parser: Parser to add the `info` command to
        parent: Parser parent of the newly added subcommand.
            used to inherit shared commands/flags

    Returns:
        The newly added parser
    """
    parser_info = parser.add_parser(
        'info',
        description='Display package and citation information',
        help='Display package and citation information',
        parents=[parent],
        add_help=False,
    )
    return parser_info


def setup_compile_args(
    parser: argparse.ArgumentParser, parent: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Helper function to set up a subparser for the `compile` command.

    Args:
        parser: Parser to add the `compile` command to
        parent: Parser parent of the newly added subcommand.
            used to inherit shared commands/flags

    Returns:
        The newly added parser
    """
    parser_compile = parser.add_parser(
        'compile',
        description='Compile `kallisto` and `bustools` binaries from source',
        help='Compile `kallisto` and `bustools` binaries from source',
        parents=[parent],
        add_help=False,
    )

    parser_compile.add_argument(
        'target',
        metavar='target',
        help=(
            'Which binaries to compile. May be one of `kallisto`, `bustools` or '
            '`all`.'
        ),
        choices=['kallisto', 'bustools', 'all'],
        default=None,
        nargs='?',
    )
    compile_group = parser_compile.add_mutually_exclusive_group()
    compile_group.add_argument(
        '--view',
        help=(
            'See information about the current binaries, which are what will be '
            'used for `ref` and `count`.'
        ),
        action='store_true',
    )
    compile_group.add_argument(
        '--remove',
        help=(
            'Remove the existing compiled binaries. Binaries that are provided '
            'with kb are never removed.'
        ),
        action='store_true',
    )
    compile_group.add_argument(
        '--overwrite',
        help='Overwrite the existing compiled binaries, if they exist.',
        action='store_true',
    )

    parser_compile.add_argument(
        '-o',
        metavar='OUT',
        help=(
            'Save the compiled binaries to a different directory. Note that if this '
            'option is specified, the binaries will have to be manually specified '
            'with `--kallisto` or `--bustools` when running `ref` or `count`.'
        ),
        type=str,
        default=None,
    )
    compile_group = parser_compile.add_mutually_exclusive_group()
    compile_group.add_argument(
        '--url',
        metavar='URL',
        help=(
            'Use a custom URL to a ZIP or tarball file containing the source code '
            'of the specified binary. May only be used with a single `target`.'
        ),
        type=str
    )
    compile_group.add_argument(
        '--ref',
        metavar='REF',
        help=(
            'Repository commmit hash or tag to fetch the source code from. '
            'May only be used with a single `target`.'
        ),
        type=str
    )
    parser_compile.add_argument(
        '--cmake-arguments',
        metavar='URL',
        help=(
            'Additional arguments to pass to the cmake command. For example, to '
            'pass additional include directories, '
            '`--cmake-arguments="-DCMAKE_CXX_FLAGS=\'-I /usr/include\'"`'
        ),
        type=str,
    )

    return parser_compile


def setup_ref_args(
    parser: argparse.ArgumentParser, parent: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Helper function to set up a subparser for the `ref` command.

    Args:
        parser: Parser to add the `ref` command to
        parent: Parser parent of the newly added subcommand.
            used to inherit shared commands/flags

    Returns:
        The newly added parser
    """
    kallisto_path = get_kallisto_binary_path()
    bustools_path = get_bustools_binary_path()

    workflow = 'standard'
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--workflow'):
            if '=' in arg:
                workflow = arg[arg.index('=') + 1:].strip('\'\"')
            else:
                workflow = sys.argv[i + 1]
            break

    parser_ref = parser.add_parser(
        'ref',
        description='Build a kallisto index and transcript-to-gene mapping',
        help='Build a kallisto index and transcript-to-gene mapping',
        parents=[parent],
    )
    parser_ref._actions[0].help = parser_ref._actions[0].help.capitalize()

    required_ref = parser_ref.add_argument_group('required arguments')
    required_ref.add_argument(
        '-i',
        metavar='INDEX',
        help='Path to the kallisto index to be constructed.',
        type=str,
        required=True
    )
    required_ref.add_argument(
        '-g',
        metavar='T2G',
        help='Path to transcript-to-gene mapping to be generated',
        type=str,
        required=workflow not in {'custom'}
    )
    required_ref.add_argument(
        '-f1',
        metavar='FASTA',
        help=(
            '[Optional with -d] Path to the cDNA FASTA (standard, nac) or '
            'mismatch FASTA (kite) to be generated '
            '[Optional with --aa when no GTF file(s) provided] '
            '[Not used with --workflow=custom]'
        ),
        type=str,
        required='-d' not in sys.argv and '--aa' not in sys.argv
        and workflow not in {'custom'}
    )
    filter_group = parser_ref.add_mutually_exclusive_group()
    filter_group.add_argument(
        '--include-attribute',
        metavar='KEY:VALUE',
        help=(
            'Only process GTF entries that have the provided KEY:VALUE attribute. '
            'May be specified multiple times.'
        ),
        type=str,
        action='append',
    )
    filter_group.add_argument(
        '--exclude-attribute',
        metavar='KEY:VALUE',
        help=(
            'Only process GTF entires that do not have the provided KEY:VALUE attribute. '
            'May be specified multiple times.'
        ),
        type=str,
        action='append',
    )

    required_nac = parser_ref.add_argument_group(
        'required arguments for `nac` workflow'
    )
    required_nac.add_argument(
        '-f2',
        metavar='FASTA',
        help='Path to the unprocessed transcripts FASTA to be generated',
        type=str,
        required=workflow in {'nac'} and '-d' not in sys.argv
    )
    required_nac.add_argument(
        '-c1',
        metavar='T2C',
        help='Path to generate cDNA transcripts-to-capture',
        type=str,
        required=workflow in {'nac'}
    )
    required_nac.add_argument(
        '-c2',
        metavar='T2C',
        help='Path to generate unprocessed transcripts-to-capture',
        type=str,
        required=workflow in {'nac'}
    )

    parser_ref.add_argument(
        '-d',
        metavar='NAME',
        help=(
            'Download a pre-built kallisto index (along with all necessary files) '
            'instead of building it locally'
        ),
        type=str,
        default=None,
        required=False
    )
    parser_ref.add_argument(
        '-k',
        metavar='K',
        help=(
            'Use this option to override the k-mer length of the index. '
            'Usually, the k-mer length automatically calculated by `kb` provides '
            'the best results.'
        ),
        type=int,
        default=None,
        required=False
    )
    parser_ref.add_argument(
        '-t',
        metavar='THREADS',
        help=('Number of threads to use (default: 8)'),
        type=int,
        default=8
    )
    parser_ref.add_argument(
        '--d-list',
        metavar='FASTA',
        help=(
            'D-list file(s) (default: the Genomic FASTA file(s) for standard/nac workflow)'
        ),
        type=str,
        default=None
    )
    parser_ref.add_argument(
        '--d-list-overhang', help=argparse.SUPPRESS, type=int, default=1
    )
    parser_ref.add_argument(
        '--aa',
        help='Generate index from a FASTA-file containing amino acid sequences',
        action='store_true',
        default=False
    )
    parser_ref.add_argument(
        '--workflow',
        metavar='{standard,nac,kite,custom}',
        help=(
            'The type of index to create. '
            'Use `nac` for an index type that can quantify nascent and mature RNA. '
            'Use `custom` for indexing targets directly. '
            'Use `kite` for feature barcoding. (default: standard)'
        ),
        type=str,
        default='standard',
        choices=['standard', 'nac', 'kite', 'custom', 'lamanno', 'nucleus']
    )
    parser_ref.add_argument(
        '--distinguish', help=argparse.SUPPRESS, action='store_true'
    )
    parser_ref.add_argument(
        '--make-unique',
        help='Replace repeated target names with unique names',
        action='store_true'
    )
    parser_ref.add_argument(
        '--overwrite',
        help='Overwrite existing kallisto index',
        action='store_true'
    )
    parser_ref.add_argument(
        '--kallisto',
        help=f'Path to kallisto binary to use (default: {kallisto_path})',
        type=str,
        default=kallisto_path
    )
    parser_ref.add_argument(
        '--bustools',
        help=f'Path to bustools binary to use (default: {bustools_path})',
        type=str,
        default=bustools_path
    )
    parser_ref.add_argument(
        '--opt-off',
        help='Disable performance optimizations',
        action='store_true'
    )
    parser_ref.add_argument(
        'fasta',
        help='Genomic FASTA file(s), comma-delimited',
        type=str,
        nargs=None if '-d' not in sys.argv and workflow != 'kite' else '?'
    )
    parser_ref.add_argument(
        'gtf',
        help='Reference GTF file(s), comma-delimited [not required with --aa]',
        type=str,
        nargs=None if ('-d' not in sys.argv and '--aa' not in sys.argv)
        and workflow not in {'custom', 'kite'} else '?'
    )
    parser_ref.add_argument(
        'feature',
        help=(
            '[`kite` workflow only] Path to TSV containing barcodes and feature names.'
        ),
        type=str,
        nargs=None if '-d' not in sys.argv and workflow == 'kite' else '?'
    )

    # Hidden options.
    parser_ref.add_argument(
        '--no-mismatches', help=argparse.SUPPRESS, action='store_true'
    )
    parser_ref.add_argument(
        '--ec-max-size', help=argparse.SUPPRESS, type=int, default=None
    )
    parser_ref.add_argument('--flank', help=argparse.SUPPRESS, type=int)

    return parser_ref


def setup_count_args(
    parser: argparse.ArgumentParser, parent: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Helper function to set up a subparser for the `count` command.

    Args:
        parser: Parser to add the `count` command to
        parent: Parser parent of the newly added subcommand.
            used to inherit shared commands/flags

    Returns:
        The newly added parser
    """
    kallisto_path = get_kallisto_binary_path()
    bustools_path = get_bustools_binary_path()

    workflow = 'standard'
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--workflow'):
            if '=' in arg:
                workflow = arg[arg.index('=') + 1:].strip('\'\"')
            else:
                workflow = sys.argv[i + 1]
            break

    # count
    parser_count = parser.add_parser(
        'count',
        description=('Generate count matrices from a set of single-cell FASTQ files. '
                     'Run `kb --list` to view single-cell technology information.'),  # noqa
        help='Generate count matrices from a set of single-cell FASTQ files',
        parents=[parent],
    )
    parser_count._actions[0].help = parser_count._actions[0].help.capitalize()

    required_count = parser_count.add_argument_group('required arguments')
    required_count.add_argument(
        '-i',
        metavar='INDEX',
        help='Path to kallisto index',
        type=str,
        required=True
    )
    required_count.add_argument(
        '-g',
        metavar='T2G',
        help='Path to transcript-to-gene mapping',
        type=str,
        required=True
    )
    required_count.add_argument(
        '-x',
        metavar='TECHNOLOGY',
        help='Single-cell technology used (`kb --list` to view)',
        type=str,
        required=True,
    )
    parser_count.add_argument(
        '-o',
        metavar='OUT',
        help='Path to output directory (default: current directory)',
        type=str,
        default='.',
    )
    parser_count.add_argument(
        '--num', help='Store read numbers in BUS file', action='store_true'
    )
    parser_count.add_argument(
        '-w',
        metavar='ONLIST',
        help=(
            'Path to file of on-listed barcodes to correct to. '
            'If not provided and bustools supports the technology, '
            'a pre-packaged on-list is used. Otherwise, '
            'the bustools allowlist command is used. '
            'Specify NONE to bypass barcode error correction. '
            '(`kb --list` to view on-lists)'
        ),
        type=str
    )
    parser_count.add_argument(
        '--exact-barcodes',
        help=('Only exact matches are used for matching barcodes to on-list.'),
        action='store_true'
    )
    parser_count.add_argument(
        '-r',
        metavar='REPLACEMENT',
        help=(
            'Path to file of a replacement list to correct to. '
            'In the file, the first column is the original barcode and second is the replacement sequence'
        ),
        type=str,
        default=None
    )
    parser_count.add_argument(
        '-t',
        metavar='THREADS',
        help='Number of threads to use (default: 8)',
        type=int,
        default=8
    )
    parser_count.add_argument(
        '-m',
        metavar='MEMORY',
        help='Maximum memory used (default: 2G for standard, 4G for others)',
        type=str,
        default='2G' if workflow == 'standard' else '4G'
    )
    parser_count.add_argument(
        '--strand',
        help='Strandedness (default: see `kb --list`)',
        type=str,
        default=None,
        choices=['unstranded', 'forward', 'reverse']
    )
    parser_count.add_argument(
        '--inleaved',
        help='Specifies that input is an interleaved FASTQ file',
        action='store_true'
    )
    parser_count.add_argument(
        '--genomebam',
        help=argparse.SUPPRESS,
        action='store_true',
        default=False,
    )
    parser_count.add_argument(
        '--aa',
        help=(
            'Map to index generated from FASTA-file containing '
            'amino acid sequences'
        ),
        action='store_true',
        default=False
    )
    parser_count.add_argument(
        '--gtf',
        help=argparse.SUPPRESS,
        type=str,
        default=None,
    )
    parser_count.add_argument(
        '--chromosomes',
        metavar='chrom.sizes',
        help=argparse.SUPPRESS,
        type=str,
        default=None,
    )
    parser_count.add_argument(
        '--workflow',
        metavar='{standard,nac,kite,kite:10xFB}',
        help=(
            'Type of workflow. '
            'Use `nac` to specify a nac index for producing mature/nascent/ambiguous matrices. '
            'Use `kite` for feature barcoding. '
            'Use `kite:10xFB` for 10x Genomics Feature Barcoding technology. '
            '(default: standard)'
        ),
        type=str,
        default='standard',
        choices=['standard', 'nac', 'kite', 'kite:10xFB', 'lamanno', 'nucleus']
    )
    parser_count.add_argument(
        '--em', help=argparse.SUPPRESS, action='store_true'
    )

    count_group = parser_count.add_mutually_exclusive_group()
    count_group.add_argument(
        '--mm',
        help=(
            'Include reads that pseudoalign to multiple genes. '
            'Automatically enabled when generating a TCC matrix.'
        ),
        action='store_true'
    )
    count_group.add_argument(
        '--tcc',
        help='Generate a TCC matrix instead of a gene count matrix.',
        action='store_true'
    )
    parser_count.add_argument(
        '--filter',
        help='Produce a filtered gene count matrix (default: bustools)',
        type=str,
        const='bustools',
        nargs='?',
        choices=['bustools']
    )
    parser_count.add_argument(
        '--filter-threshold',
        metavar='THRESH',
        help='Barcode filter threshold (default: auto)',
        type=int,
        default=None,
    )
    required_nac = parser_count.add_argument_group(
        'required arguments for `nac` workflow'
    )
    required_nac.add_argument(
        '-c1',
        metavar='T2C',
        help='Path to mature transcripts-to-capture',
        type=str,
        required=workflow in {'nac'}
    )
    required_nac.add_argument(
        '-c2',
        metavar='T2C',
        help='Path to nascent transcripts-to-captured',
        type=str,
        required=workflow in {'nac'}
    )
    parser_count.add_argument(
        '--overwrite',
        help='Overwrite existing output.bus file',
        action='store_true'
    )
    parser_count.add_argument('--dry-run', help='Dry run', action='store_true')
    parser_count.add_argument(
        '--batch-barcodes',
        help=(
            'When a batch file is supplied, store sample identifiers '
            'in barcodes'
        ),
        action='store_true'
    )

    conversion_group = parser_count.add_mutually_exclusive_group()
    conversion_group.add_argument(
        '--loom',
        help='Generate loom file from count matrix',
        action='store_true'
    )
    conversion_group.add_argument(
        '--h5ad',
        help='Generate h5ad file from count matrix',
        action='store_true'
    )
    parser_count.add_argument(
        '--loom-names',
        metavar='col_attrs/{name},row_attrs/{name}',
        help=(
            'Names for col_attrs and row_attrs in loom file (default: barcode,target_name). '
            'Use --loom-names=velocyto for velocyto-compatible loom files'
        ),
        type=str,
        default="barcode,target_name",
    )
    parser_count.add_argument(
        '--sum',
        metavar='TYPE',
        help=(
            'Produced summed count matrices (Options: none, cell, nucleus, total). '
            'Use `cell` to add ambiguous and processed transcript matrices. '
            'Use `nucleus` to add ambiguous and unprocessed transcript matrices. '
            'Use `total` to add all three matrices together. '
            '(Default: none)'
        ),
        type=str,
        default="none",
        choices=['none', 'cell', 'nucleus', 'total']
    )
    parser_count.add_argument(
        '--cellranger',
        help='Convert count matrices to cellranger-compatible format',
        action='store_true'
    )
    parser_count.add_argument(
        '--gene-names',
        help=(
            'Group counts by gene names instead of gene IDs when generating '
            'the loom or h5ad file'
        ),
        action='store_true'
    )
    parser_count.add_argument(
        '-N',
        metavar='NUMREADS',
        help='Maximum number of reads to process from supplied input',
        type=int,
        default=None
    )

    report_group = parser_count.add_mutually_exclusive_group()
    report_group.add_argument(
        '--report',
        help=(
            'Generate a HTML report containing run statistics and basic plots. '
            'Using this option may cause kb to use more memory than specified '
            'with the `-m` option. It may also cause it to crash due to memory.'
        ),
        action='store_true'
    )
    report_group.add_argument(
        '--no-inspect', help=argparse.SUPPRESS, action='store_true'
    )
    parser_count.add_argument(
        '--long',
        help='Use lr-kallisto for long-read mapping',
        action='store_true'
    )
    parser_count.add_argument(
        '--threshold',
        metavar='THRESH',
        help='Set threshold for lr-kallisto read mapping (default: 0.8)',
        type=float,
        default=0.8
    )
    parser_count.add_argument(
        '--error-rate', help=argparse.SUPPRESS, type=float, default=None
    )
    parser_count.add_argument(
        '--platform',
        metavar='[PacBio or ONT]',
        help='Set platform for lr-kallisto (default: ONT)',
        type=str,
        default='ONT',
        choices=['PacBio', 'ONT']
    )
    parser_count.add_argument(
        '--kallisto',
        help=f'Path to kallisto binary to use (default: {kallisto_path})',
        type=str,
        default=kallisto_path
    )
    parser_count.add_argument(
        '--bustools',
        help=f'Path to bustools binary to use (default: {bustools_path})',
        type=str,
        default=bustools_path
    )
    parser_count.add_argument(
        '--opt-off',
        help='Disable performance optimizations',
        action='store_true'
    )
    parser_count.add_argument(
        '-k', help=argparse.SUPPRESS, type=int, default=31
    )
    parser_count.add_argument(
        '--no-validate', help=argparse.SUPPRESS, action='store_true'
    )
    parser_count.add_argument(
        '--no-fragment', help=argparse.SUPPRESS, action='store_true'
    )
    parser_count.add_argument(
        '--union', help=argparse.SUPPRESS, action='store_true'
    )
    parser_count.add_argument(
        '--no-jump', help=argparse.SUPPRESS, action='store_true'
    )
    parser_count.add_argument(
        '--quant-umis', help=argparse.SUPPRESS, action='store_true'
    )
    parser_count.add_argument(
        '--keep-flags', help=argparse.SUPPRESS, action='store_true'
    )

    optional_bulk = parser_count.add_argument_group(
        'optional arguments for `BULK` and `SMARTSEQ2` technologies'
    )
    optional_bulk.add_argument(
        '--parity',
        help=(
            'Parity of the input files. Choices are `single` for single-end '
            'and `paired` for paired-end reads.'
        ),
        type=str,
        choices=['single', 'paired'],
        default=None
    )
    optional_bulk.add_argument(
        '--fragment-l',
        metavar='L',
        help='Mean length of fragments. Only for single-end.',
        type=int,
        default=None
    )
    optional_bulk.add_argument(
        '--fragment-s',
        metavar='S',
        help='Standard deviation of fragment lengths. Only for single-end.',
        type=int,
        default=None
    )
    optional_bulk.add_argument(
        '--bootstraps',
        metavar='B',
        help='Number of bootstraps to perform',
        type=int,
        default=None
    )
    optional_bulk.add_argument(
        '--matrix-to-files',
        help='Reorganize matrix output into abundance tsv files',
        action='store_true'
    )
    optional_bulk.add_argument(
        '--matrix-to-directories',
        help=(
            'Reorganize matrix output into abundance tsv files across '
            'multiple directories'
        ),
        action='store_true'
    )

    parser_count.add_argument(
        'fastqs',
        help=(
            'FASTQ files. For technology `SMARTSEQ`, all input FASTQs are '
            'alphabetically sorted by path and paired in order, and cell IDs '
            'are assigned as incrementing integers starting from zero. A single '
            'batch TSV with cell ID, read 1, and read 2 as columns can be '
            'provided to override this behavior.'
        ),
        nargs='+'
    )
    return parser_count


def setup_extract_args(
    parser: argparse.ArgumentParser, parent: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    """Helper function to set up a subparser for the `extract` command.

    Args:
        parser: Parser to add the `extract` command to
        parent: Parser parent of the newly added subcommand.
            used to inherit shared commands/flags

    Returns:
        The newly added parser
    """
    kallisto_path = get_kallisto_binary_path()
    bustools_path = get_bustools_binary_path()

    parser_extract = parser.add_parser(
        'extract',
        description=(
            'Extract sequencing reads that were pseudoaligned to specific genes/transcripts '
            '(or extract all reads that were / were not pseudoaligned).'
        ),
        help=(
            'Extract sequencing reads that were pseudoaligned to specific genes/transcripts '
            '(or extract all reads that were / were not pseudoaligned)'
        ),
        parents=[parent]
    )
    parser_extract._actions[0].help = parser_extract._actions[
        0].help.capitalize()

    required_extract = parser_extract.add_argument_group('required arguments')
    required_extract.add_argument(
        'fastq',
        metavar='FASTQ',
        type=str,
        help=(
            'Single fastq file containing the sequencing reads (e.g. in case of 10x data, provide the R2 file).'
            ' Sequencing technology will be treated as bulk here since barcode and UMI tracking '
            'is not necessary to extract reads.'
        )
    )
    required_extract.add_argument(
        '-i',
        metavar='INDEX',
        type=str,
        required=True,
        help='Path to kallisto index'
    )
    required_extract.add_argument(
        '-ts',
        '--targets',
        metavar='TARGETS',
        type=str,
        nargs='+',
        required=False,
        default=None,
        help=(
            'Gene or transcript names for which to extract the raw reads that align to the index'
        )
    )
    parser_extract.add_argument(
        '-ttype',
        '--target_type',
        metavar='TYPE',
        type=str,
        default='gene',
        choices=['gene', 'transcript'],
        help=(
            "'gene' (default) or 'transcript' -> Defines whether targets are gene or transcript names"
        )
    )
    parser_extract.add_argument(
        '--extract_all',
        help=(
            'Extracts all reads that pseudo-aligned to any gene or transcript (as defined by target_type) '
            '(breaks down output by gene/transcript). '
            'Using extract_all might take a long time to run when there are a large number of '
            'genes/transcripts in the index.'
        ),
        action='store_true',
        default=False
    )
    parser_extract.add_argument(
        '--extract_all_fast',
        help=(
            'Extracts all reads that pseudo-aligned (does not break down output by gene/transcript; '
            'output saved in the "all" folder).'
        ),
        action='store_true',
        default=False
    )
    parser_extract.add_argument(
        '--extract_all_unmapped',
        help=(
            'Extracts all unmapped reads (output saved in the "all_unmapped" folder).'
        ),
        action='store_true',
        default=False
    )
    parser_extract.add_argument(
        '--mm',
        help=('Also extract reads that multi-mapped to more than one gene.'),
        action='store_true',
        default=False
    )
    parser_extract.add_argument(
        '-g',
        metavar='T2G',
        help=(
            'Path to transcript-to-gene mapping file '
            '(required when mm = False, target_type = "gene" '
            '(and extract_all_fast and extract_all_unmapped = False), OR extract_all = True).'
        ),
        type=str,
    )
    parser_extract.add_argument(
        '-o',
        metavar='OUT',
        help='Path to output directory (default: current directory)',
        type=str,
        default='.',
    )
    parser_extract.add_argument(
        '-t',
        metavar='THREADS',
        help='Number of threads to use (default: 8)',
        type=int,
        default=8
    )
    parser_extract.add_argument(
        '-s',
        '--strand',
        help="Strandedness (default: 'unstranded')",
        type=str,
        default=None,
        choices=['unstranded', 'forward', 'reverse']
    )
    parser_extract.add_argument(
        '--aa',
        help=(
            'Map to index generated from FASTA-file'
            ' containing amino acid sequences'
        ),
        action='store_true',
        default=False
    )
    parser_extract.add_argument(
        '-N',
        metavar='NUMREADS',
        help='Maximum number of reads to process from supplied fastq',
        type=int,
        default=None
    )
    parser_extract.add_argument(
        '--kallisto',
        help=f'Path to kallisto binary to use (default: {kallisto_path})',
        type=str,
        default=kallisto_path
    )
    parser_extract.add_argument(
        '--bustools',
        help=f'Path to bustools binary to use (default: {bustools_path})',
        type=str,
        default=bustools_path
    )
    parser_extract.add_argument(
        '--opt-off',
        help='Disable performance optimizations',
        action='store_true'
    )
    parser_extract.add_argument(
        '-k', help=argparse.SUPPRESS, type=int, default=31
    )

    return parser_extract


@logger.namespaced('main')
def main():
    """Command-line entrypoint.
    """
    # Main parser
    parser = argparse.ArgumentParser(
        description='kb_python {}'.format(__version__)
    )
    parser._actions[0].help = parser._actions[0].help.capitalize()
    parser.add_argument(
        '--list',
        help='Display list of supported single-cell technologies',
        action='store_true'
    )
    subparsers = parser.add_subparsers(
        dest='command',
        metavar='<CMD>',
    )

    # Add common options to this parent parser
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        '--tmp',
        metavar='TMP',
        help='Override default temporary directory',
        type=str
    )
    parent.add_argument(
        '--keep-tmp',
        help='Do not delete the tmp directory',
        action='store_true'
    )
    parent.add_argument(
        '--verbose', help='Print debugging information', action='store_true'
    )

    # Command parsers
    setup_info_args(subparsers, argparse.ArgumentParser(add_help=False))
    parser_compile = setup_compile_args(subparsers, parent)
    parser_ref = setup_ref_args(subparsers, parent)
    parser_count = setup_count_args(subparsers, parent)
    parser_extract = setup_extract_args(subparsers, parent)

    command_to_parser = {
        'compile': parser_compile,
        'ref': parser_ref,
        'count': parser_count,
        'extract': parser_extract,
    }
    if 'info' in sys.argv:
        display_info()
    elif '--list' in sys.argv:
        display_technologies()

    # Show help when no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if len(sys.argv) == 2:
        if sys.argv[1] in command_to_parser:
            command_to_parser[sys.argv[1]].print_help(sys.stderr)
        else:
            parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Validation
    if 'no_validate' in args and args.no_validate:
        logger.warning((
            'File validation is turned off. '
            'This may lead to corrupt/empty output files.'
        ))
        no_validate()

    if 'dry_run' in args:
        # Dry run can not be specified with matrix conversion.
        if args.dry_run and (args.loom or args.h5ad or args.cellranger
                             or args.gene_names):
            raise parser.error(
                '--dry-run can not be used with --loom, --h5ad, --cellranger, or --gene-names'
            )

        if args.dry_run:
            logging.disable(level=logging.CRITICAL)
            set_dry()

    logger.debug('Printing verbose output')

    # Set binary paths
    if args.command in ('ref', 'count', 'extract'):
        dry_run = not ('dry_run' not in args or not args.dry_run)
        use_kmer64 = False
        opt_off = False
        if args.k and args.k > 32:
            use_kmer64 = True
        if args.opt_off:
            opt_off = True
        # Handle larger k-mer sizes or disable optimizations
        if use_kmer64 or opt_off:
            # Only do so if --kallisto not already provided
            if not any(arg.startswith('--kallisto') for arg in sys.argv):
                set_special_kallisto_binary(use_kmer64, opt_off)
                args.kallisto = get_provided_kallisto_path()

        if args.kallisto:
            set_kallisto_binary_path(args.kallisto)
        if args.bustools:
            set_bustools_binary_path(args.bustools)

        # Check
        kallisto_path = get_kallisto_binary_path()
        bustools_path = get_bustools_binary_path()
        kallisto_ok = True
        bustools_ok = True
        if not dry_run:
            kallisto_ok, bustools_ok = test_binaries()

        # If kallisto binary is not OK, try one with opt-off if applicable
        if not kallisto_ok and not opt_off and bustools_ok:
            # Only do so if --kallisto not already provided
            if not any(arg.startswith('--kallisto') for arg in sys.argv):
                opt_off = True
                set_special_kallisto_binary(use_kmer64, opt_off)
                args.kallisto = get_provided_kallisto_path()
                set_kallisto_binary_path(args.kallisto)
                kallisto_path = get_kallisto_binary_path()
                kallisto_ok, bustools_ok = test_binaries()

        if not kallisto_path or not kallisto_ok:
            raise UnsupportedOSError(
                'Failed to find compatible kallisto binary. '
                'Provide a compatible binary with the `--kallisto` option or '
                'run `kb compile`.'
            )
        if not bustools_path or not bustools_ok:
            raise UnsupportedOSError(
                'Failed to find compatible bustools binary. '
                'Provide a compatible binary with the `--bustools` option or '
                'run `kb compile`.'
            )

        if not dry_run:
            logger.debug(f'kallisto binary located at {kallisto_path}')
            logger.debug(f'bustools binary located at {bustools_path}')

    temp_dir = args.tmp or (
        os.path.join(args.o, TEMP_DIR)
        if 'o' in args and args.o is not None else TEMP_DIR
    )
    # Check if temp_dir exists and exit if it does.
    # This is so that kb doesn't accidently use an existing directory and
    # delete it afterwards.
    if os.path.exists(temp_dir):
        parser.error(
            f'Temporary directory `{temp_dir}` exists! Is another instance running? '
            'Either remove the existing directory or use `--tmp` to specify a '
            'different temporary directory.'
        )

    logger.debug(f'Creating `{temp_dir}` directory')
    make_directory(temp_dir)
    try:
        logger.debug(args)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            COMMAND_TO_FUNCTION[args.command](parser, args, temp_dir=temp_dir)
    except Exception:
        if is_dry():
            raise
        logger.exception('An exception occurred')
    finally:
        # Always clean temp dir
        if not args.keep_tmp:
            logger.debug(f'Removing `{temp_dir}` directory')
            remove_directory(temp_dir)
