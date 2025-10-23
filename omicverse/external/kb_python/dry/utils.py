import os
import tempfile
from typing import List

import ngs_tools as ngs

from ..config import (
    PLATFORM,
    TECHNOLOGIES_MAPPING,
    UnsupportedOSError,
)


def run_executable(command: List[str], quiet: bool = False, *args, **kwargs):
    """Dry version of `utils.run_executable`.
    """
    command = [str(c) for c in command]
    if not quiet:
        c = command.copy()
        print(' '.join(c))


def make_directory(path: str):
    """Dry version of `utils.make_directory`.
    """
    if PLATFORM == 'windows':
        print('md {}'.format(path))
    else:
        print('mkdir -p {}'.format(path))


def remove_directory(path: str):
    """Dry version of `utils.remove_directory`.
    """
    if PLATFORM == 'windows':
        print('rd /s /q "{}"'.format(path))
    else:
        print('rm -rf {}'.format(path))


def stream_file(url: str, path: str) -> str:
    """Dry version of `utils.stream_file`.
    """
    if PLATFORM == 'windows':
        raise UnsupportedOSError((
            'Windows does not support piping remote files.'
            'Please download the file manually.'
        ))
    else:
        print('mkfifo {}'.format(path))
        print('wget -bq {} -O {}'.format(url, path))
        return path


def move_file(source: str, destination: str) -> str:
    """Dry version of `utils.move_file`.
    """
    if PLATFORM == 'windows':
        print(f'move {source} {destination}')
    else:
        print(f'mv {source} {destination}')
    return destination


def copy_whitelist(technology: str, out_dir: str) -> str:
    """Dry version of `utils.copy_whitelist`.
    """
    technology = TECHNOLOGIES_MAPPING[technology.upper()]
    archive_path = technology.chemistry.whitelist_path
    whitelist_path = os.path.join(
        out_dir,
        os.path.splitext(os.path.basename(archive_path))[0]
    )
    print('gzip -dc {} > {}'.format(archive_path, whitelist_path))
    return whitelist_path


def create_10x_feature_barcode_map(out_path: str) -> str:
    """Dry version of `utils.create_10x_feature_barcode_map`.
    """
    chemistry = ngs.chemistry.get_chemistry('10xFB')
    gex = chemistry.chemistry('gex')
    fb = chemistry.chemistry('fb')
    if PLATFORM == 'windows':
        raise UnsupportedOSError(
            "10x Feature Barcode dry run is not supported on Windows. "
            f"Please manually combine {fb.whitelist_path} and {gex.whitelist_path} "
            "into a TSV with the former as the first column and the latter as the second."
        )
    elif PLATFORM == 'linux':
        print(
            f'paste <(zcat {fb.whitelist_path}) <(zcat {gex.whitelist_path}) > {out_path}'
        )
    elif PLATFORM == 'darwin':
        print(
            f'paste <(gzcat {fb.whitelist_path}) <(gzcat {gex.whitelist_path}) > {out_path}'
        )
    else:
        raise UnsupportedOSError(f'Unrecognized platform {PLATFORM}')
    return out_path


def get_temporary_filename(temp_dir: str) -> str:
    """Dry version of `utils.get_temporary_filename`.
    """
    return os.path.join(
        temp_dir, f'{tempfile.gettempprefix()}{tempfile._get_candidate_names()}'
    )
