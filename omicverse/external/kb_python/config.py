import os
import platform
import shutil
from collections import namedtuple
from typing import NamedTuple, Optional
from urllib.parse import urljoin

import ngs_tools as ngs

PACKAGE_PATH = os.path.abspath(os.path.dirname(__file__))
PLATFORM = platform.system().lower()
CPU = ''
if PLATFORM == 'darwin' and platform.processor().lower() == 'arm':
    CPU = 'm1'
BINS_DIR = os.path.join(PACKAGE_PATH, 'bins')
COMPILED_DIR = os.path.join(BINS_DIR, 'compiled')

TEMP_DIR = 'tmp'
DRY = False
VALIDATE = True

GITHUB_API_URL = 'https://api.github.com'
KALLISTO_REPO_URL = urljoin(GITHUB_API_URL, 'repos/pachterlab/kallisto/')
BUSTOOLS_REPO_URL = urljoin(GITHUB_API_URL, 'repos/BUStools/bustools/')
KALLISTO_RELEASES_URL = urljoin(KALLISTO_REPO_URL, 'releases')
BUSTOOLS_RELEASES_URL = urljoin(BUSTOOLS_REPO_URL, 'releases')
KALLISTO_TARBALL_URL = urljoin(KALLISTO_REPO_URL, 'tarball/')
BUSTOOLS_TARBALL_URL = urljoin(BUSTOOLS_REPO_URL, 'tarball/')


def get_provided_kallisto_path() -> Optional[str]:
    """Finds platform-dependent kallisto binary included with the installation.

    Returns:
        Path to the binary, `None` if not found
    """
    bin_name = 'kallisto'
    if '_KALLISTO_OPTOFF' in globals():
        if _KALLISTO_OPTOFF:
            bin_name = f'{bin_name}_optoff'
    if '_KALLISTO_KMER_64' in globals():
        if _KALLISTO_KMER_64:
            bin_name = f'{bin_name}_k64'
    bin_filename = f'{bin_name}.exe' if PLATFORM == 'windows' else bin_name
    path = os.path.join(BINS_DIR, PLATFORM, CPU, 'kallisto', bin_filename)
    if not os.path.isfile(path):
        return None
    return path


def get_provided_bustools_path() -> Optional[str]:
    """Finds platform-dependent bustools binary included with the installation.

    Returns:
        Path to the binary, `None` if not found
    """
    bin_filename = 'bustools.exe' if PLATFORM == 'windows' else 'bustools'
    path = os.path.join(BINS_DIR, PLATFORM, CPU, 'bustools', bin_filename)
    if not os.path.isfile(path):
        return None
    return path


def set_special_kallisto_binary(k64: bool, optoff: bool):
    global _KALLISTO_KMER_64
    global _KALLISTO_OPTOFF
    _KALLISTO_KMER_64 = k64
    _KALLISTO_OPTOFF = optoff


def get_compiled_kallisto_path(alias: str = COMPILED_DIR) -> Optional[str]:
    """Finds platform-dependent kallisto binary compiled with `compile`.

    Args:
        alias: Alias of compiled binary.

    Returns:
        Path to the binary, `None` if not found
    """
    bin_filename = 'kallisto.exe' if PLATFORM == 'windows' else 'kallisto'
    path = os.path.join(alias, 'kallisto', bin_filename)
    if not os.path.isfile(path):
        return None
    return path


def get_compiled_bustools_path(alias: str = COMPILED_DIR) -> Optional[str]:
    """Finds platform-dependent bustools binary compiled with `compile`.

    Args:
        alias: Alias of compiled binary.

    Returns:
        Path to the binary, `None` if not found
    """
    bin_filename = 'bustools.exe' if PLATFORM == 'windows' else 'bustools'
    path = os.path.join(alias, 'bustools', bin_filename)
    if not os.path.isfile(path):
        return None
    return path


# Binary paths. These should hold the full path to the binaries that should
# be called throughout the execution of the program. Therefore, this
# usually needs to be set only once. Defaults to provided binaries.
KALLISTO_PATH = get_compiled_kallisto_path() or get_provided_kallisto_path()
BUSTOOLS_PATH = get_compiled_bustools_path() or get_provided_bustools_path()


# Technology to file position mapping
class Technology(NamedTuple):
    name: str
    description: str
    chemistry: ngs.chemistry.Chemistry
    show: bool = True


TECHNOLOGIES = sorted([
    Technology('10XV1', '10x version 1', ngs.chemistry.get_chemistry('10xv1')),
    Technology('10XV2', '10x version 2', ngs.chemistry.get_chemistry('10xv2')),
    Technology('10XV3', '10x version 3', ngs.chemistry.get_chemistry('10xv3')),
    Technology(
        '10XV3_ULTIMA', '10x version 3 sequenced with Ultima',
        ngs.chemistry.get_chemistry('10xv3_Ultima')
    ),
    Technology(
        '10XFB', '10x Feature Barcode',
        ngs.chemistry.get_chemistry('10xFBonly'), False
    ),
    Technology('10XV4', '10x version 4', ngs.chemistry.get_chemistry('10xv4')),
    Technology('CELSEQ', 'CEL-Seq', ngs.chemistry.get_chemistry('celseq')),
    Technology(
        'CELSEQ2', 'CEL-SEQ version 2', ngs.chemistry.get_chemistry('celseq2')
    ),
    Technology('DROPSEQ', 'DropSeq', ngs.chemistry.get_chemistry('dropseq')),
    Technology(
        'INDROPSV1', 'inDrops version 1',
        ngs.chemistry.get_chemistry('indropsv1')
    ),
    Technology(
        'INDROPSV2', 'inDrops version 2',
        ngs.chemistry.get_chemistry('indropsv2')
    ),
    Technology(
        'INDROPSV3', 'inDrops version 3',
        ngs.chemistry.get_chemistry('indropsv3')
    ),
    Technology('SCRUBSEQ', 'SCRB-Seq', ngs.chemistry.get_chemistry('scrbseq')),
    Technology(
        'SURECELL', 'SureCell for ddSEQ',
        ngs.chemistry.get_chemistry('surecell')
    ),
    Technology(
        'SMARTSEQ2', 'Smart-seq2  (single or paired)',
        ngs.chemistry.get_chemistry('smartseq2')
    ),
    Technology(
        'BULK', 'Bulk (single or paired)',
        ngs.chemistry.get_chemistry('smartseq2')
    ),
    Technology(
        'SMARTSEQ3', 'Smart-seq3', ngs.chemistry.get_chemistry('smartseq3')
    ),
    Technology(
        'STORMSEQ', 'STORM-seq', ngs.chemistry.get_chemistry('stormseq')
    ),
    Technology(
        'BDWTA', 'BD Rhapsody', ngs.chemistry.get_chemistry('bd rhapsody')
    ),
    Technology('Visium', '10x Visium', ngs.chemistry.get_chemistry('visium')),
    Technology(
        'SPLIT-SEQ', 'SPLiT-seq (version 2)',
        ngs.chemistry.get_chemistry('split-seq')
    ),
],
                      key=lambda t: t.name)
TECHNOLOGIES_MAPPING = {t.name: t for t in TECHNOLOGIES}

# Supported pre-built indices
Reference = namedtuple('Reference', ['name', 'url', 'files'])
REFERENCES = [
    Reference(
        'human',
        'https://caltech.box.com/shared/static/v1nm7lpnqz5syh8dyzdk2zs8bglncfib.gz',
        {
            'i': 'transcriptome.idx',
            'g': 'transcripts_to_genes.txt'
        }
    ),
    Reference(
        'mouse',
        'https://caltech.box.com/shared/static/vcaz6cujop0xuapdmz0pplp3aoqc41si.gz',
        {
            'i': 'transcriptome.idx',
            'g': 'transcripts_to_genes.txt'
        }
    ),
    Reference(
        'linnarsson',
        'https://caltech.box.com/shared/static/kyf7ai5s8y2l0vycl5yxunrappvrf0yx.gz',
        {
            'i': 'gencode.v31.fragments.idx',
            'g': 'fragments2genes.txt',
            'c1': 'spliced_fragments.txt',
            'c2': 'unspliced_fragments.txt',
        }
    )
]
REFERENCES_MAPPING = {r.name: r for r in REFERENCES}


class UnsupportedOSError(Exception):
    pass


class ConfigError(Exception):
    pass


def get_kallisto_binary_path() -> str:
    """Dummy function that simply returns the current value of :data:`KALLISTO_PATH`.
    """
    return KALLISTO_PATH


def get_bustools_binary_path() -> str:
    """Dummy function that simply returns the current value of :data:`BUSTOOLS_PATH`.
    """
    return BUSTOOLS_PATH


def set_kallisto_binary_path(path: str):
    """Helper function to set the :data:`KALLISTO_PATH` variable. Automatically
    finds the full path to the executable and sets that as :data:`KALLISTO_PATH`.

    Args:
        path: Path to the kallisto binary

    Raises:
        ConfigError: If `path` could not be resolved or if the executable is
            not executable.
    """
    global KALLISTO_PATH

    shutil_path = shutil.which(path)
    actual_path = None

    # First, check if it is an executable in the user's PATH
    if shutil_path:
        actual_path = os.path.abspath(shutil_path)
    elif os.path.isfile(path):
        actual_path = os.path.abspath(path)
    else:
        raise ConfigError(f'Unable to resolve path {path}')

    # Check that it is executable
    if not os.access(actual_path, os.X_OK):
        raise ConfigError(f'{actual_path} is not executable')

    KALLISTO_PATH = actual_path


def set_bustools_binary_path(path: str):
    """Helper function to set the :data:`BUSTOOLS_PATH` variable. Automatically
    finds the full path to the executable and sets that as :data:`BUSTOOLS_PATH`.

    Args:
        path: Path to the bustools binary

    Raises:
        ConfigError: If `path` could not be resolved or if the executable is
            not executable.
    """
    global BUSTOOLS_PATH

    shutil_path = shutil.which(path)
    actual_path = None

    # First, check if it is an executable in the user's PATH
    if shutil_path:
        actual_path = os.path.abspath(shutil_path)
    elif os.path.isfile(path):
        actual_path = os.path.abspath(path)
    else:
        raise ConfigError(f'Unable to resolve path {path}')

    # Check that it is executable
    if not os.access(actual_path, os.X_OK):
        raise ConfigError(f'{actual_path} is not executable')

    BUSTOOLS_PATH = actual_path


def set_dry():
    """Set this run to be a dry run.
    """
    global DRY
    DRY = True


def is_dry() -> bool:
    """Return whether the current run is a dry run.

    Returns:
        Whether the current run is a dry run
    """
    return DRY


def no_validate():
    """Turn off validation.
    """
    global VALIDATE
    VALIDATE = False


def is_validate() -> bool:
    """Return whether validation is turned on.

    Returns:
        Whether validation is on
    """
    return VALIDATE
