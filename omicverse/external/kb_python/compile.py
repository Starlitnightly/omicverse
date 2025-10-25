import os
import shutil
import tempfile
from typing import Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from typing_extensions import Literal

from .config import (
    BUSTOOLS_RELEASES_URL,
    BUSTOOLS_TARBALL_URL,
    COMPILED_DIR,
    KALLISTO_RELEASES_URL,
    KALLISTO_TARBALL_URL,
    PLATFORM,
)
from .logging import logger
from .utils import download_file, restore_cwd, run_executable


class CompileError(Exception):
    pass


def get_latest_github_release_tag(releases_url: str) -> str:
    """Get the tag name of the latest GitHub release, given a url to the
    releases API.

    Args:
        releases_url: Url to the releases API

    Returns:
        The tag name
    """
    response = requests.get(releases_url)
    response.raise_for_status()
    return response.json()[0]['tag_name']


def get_filename_from_url(url: str) -> str:
    """Fetch the filename from a URL.

    Args:
        url: The url

    Returns:
        The filename
    """
    response = requests.get(url)
    response.raise_for_status()
    disposition = response.headers.get('content-disposition')
    if disposition:
        for split in disposition.split(';'):
            split = split.strip()
            if split.startswith('filename'):
                return split[split.index('=') + 1:].strip('\"\'')
    else:
        return os.path.basename(urlparse(url).path)


def get_kallisto_url(ref: Optional[str] = None) -> str:
    """Get the tarball url of the specified or latest kallisto release.

    Args:
        ref: Commit or release tag, defaults to `None`. By default, the most
            recent release is used.

    Returns:
        Tarball url
    """
    tag = ref or get_latest_github_release_tag(KALLISTO_RELEASES_URL)
    return urljoin(KALLISTO_TARBALL_URL, tag)


def get_bustools_url(ref: Optional[str] = None) -> str:
    """Get the tarball url of the specified or latest bustools release.

    Args:
        ref: Commit or release tag, defaults to `None`. By default, the most
            recent release is used.

    Returns:
        Tarball url
    """
    tag = ref or get_latest_github_release_tag(BUSTOOLS_RELEASES_URL)
    return urljoin(BUSTOOLS_TARBALL_URL, tag)


def find_git_root(path: str) -> str:
    """Find the root directory of a git repo by walking.

    Args:
        path: Path to start the search

    Returns:
        Path to root of git repo

    Raises:
        CompileError: If the git root could not be found
    """
    for root, dirs, files in os.walk(path):
        if '.gitignore' in files:
            return root
    raise CompileError('Unable to find git root.')


@restore_cwd
def compile_kallisto(
    source_dir: str,
    binary_path: str,
    cmake_arguments: Optional[str] = None
) -> str:
    """Compile `kallisto` from source.

    Args:
        source_dir: Path to directory containing root of kallisto git repo
        binary_path: Path to place compiled binary
        cmake_arguments: Additional arguments to pass to the cmake command

    Returns:
        Path to compiled binary
    """
    source_dir = os.path.abspath(source_dir)
    binary_path = os.path.abspath(binary_path)
    os.makedirs(os.path.dirname(binary_path), exist_ok=True)

    logger.info(
        f'Compiling `kallisto` binary from source at {source_dir} to {binary_path}. '
        'This requires `autoheader`, `autoconf`, `cmake` and `make` to be executable '
        'from the command-line, as well as zlib development headers. '
        'See https://pachterlab.github.io/kallisto/source for more information.'
    )
    os.chdir(source_dir)
    shutil.copyfile(
        'license.txt',
        os.path.join(os.path.dirname(binary_path), 'license.txt')
    )

    os.chdir(os.path.join('ext', 'htslib'))
    run_executable(['autoheader'])
    run_executable(['autoconf'])
    os.chdir(os.path.join('..', '..'))
    os.makedirs('build', exist_ok=True)
    os.chdir('build')
    cmake_command = ['cmake', '..']
    if cmake_arguments:
        cmake_command.append(cmake_arguments)
    run_executable(cmake_command)
    run_executable(['make'])
    os.makedirs(os.path.dirname(binary_path), exist_ok=True)
    shutil.copy2(
        os.path.join(
            'src', 'kallisto.exe' if PLATFORM == 'windows' else 'kallisto'
        ), binary_path
    )
    return binary_path


@restore_cwd
def compile_bustools(
    source_dir: str,
    binary_path: str,
    cmake_arguments: Optional[str] = None
) -> str:
    """Compile `bustools` from source.

    Args:
        source_dir: Path to directory containing root of bustools git repo
        binary_path: Path to place compiled binary
        cmake_arguments: Additional arguments to pass to the cmake command

    Returns:
        Path to compiled binary
    """
    source_dir = os.path.abspath(source_dir)
    binary_path = os.path.abspath(binary_path)
    os.makedirs(os.path.dirname(binary_path), exist_ok=True)

    logger.info(
        f'Compiling `bustools` binary from source {source_dir} to {binary_path}. '
        'This requires `cmake` and `make` to be executable from the command-line. '
        'See https://bustools.github.io/source for more information.'
    )
    os.chdir(source_dir)
    shutil.copyfile(
        'LICENSE', os.path.join(os.path.dirname(binary_path), 'LICENSE')
    )

    os.makedirs('build', exist_ok=True)
    os.chdir('build')
    cmake_command = ['cmake', '..']
    if cmake_arguments:
        cmake_command.append(cmake_arguments)
    run_executable(cmake_command)
    run_executable(['make'])
    shutil.copy2(
        os.path.join(
            'src', 'bustools.exe' if PLATFORM == 'windows' else 'bustools'
        ), binary_path
    )
    return binary_path


@logger.namespaced('compile')
def compile(
    target: Literal['kallisto', 'bustools', 'all'],
    out_dir: Optional[str] = None,
    cmake_arguments: Optional[str] = None,
    url: Optional[str] = None,
    ref: Optional[str] = None,
    overwrite: bool = False,
    temp_dir: str = 'tmp',
) -> Dict[str, str]:
    """Compile `kallisto` and/or `bustools` binaries by downloading and compiling
    a source archive.

    Args:
        target: Which binary to compile. May be one of `kallisto`, `bustools`
            or `all`
        out_dir: Path to output directory, defaults to `None`
        cmake_arguments: Additional arguments to pass to the cmake command
        url: Download the source archive from this url instead, defaults to
            `None`
        ref: Commit hash or tag to use, defaults to `None`
        overwrite: Overwrite any existing results, defaults to `False`
        temp_dir: Path to temporary directory, defaults to `tmp`

    Returns:
        Dictionary of results
    """
    results = {}
    if target in ('kallisto', 'all'):
        binary_path = os.path.join(
            out_dir or os.path.join(COMPILED_DIR, 'kallisto'), 'kallisto'
        )
        if os.path.exists(binary_path) and not overwrite:
            raise Exception(
                f'Compiled binary already exists at {binary_path}. '
                'Use `--overwrite` to overwrite.'
            )

        _url = url or get_kallisto_url(ref)
        logger.info(f'Downloading kallisto source from {_url}')
        archive_path = download_file(
            _url, os.path.join(temp_dir, get_filename_from_url(_url))
        )
        source_dir = tempfile.mkdtemp(dir=temp_dir)
        shutil.unpack_archive(archive_path, source_dir)
        source_dir = find_git_root(source_dir)
        binary_path = compile_kallisto(
            source_dir, binary_path, cmake_arguments=cmake_arguments
        )
        results['kallisto'] = binary_path
    if target in ('bustools', 'all'):
        binary_path = os.path.join(
            out_dir or os.path.join(COMPILED_DIR, 'bustools'), 'bustools'
        )
        if os.path.exists(binary_path) and not overwrite:
            raise Exception(
                f'Compiled binary already exists at {binary_path}. '
                'Use `--overwrite` to overwrite.'
            )

        _url = url or get_bustools_url(ref)
        logger.info(f'Downloading bustools source from {_url}')
        archive_path = download_file(
            _url, os.path.join(temp_dir, get_filename_from_url(_url))
        )
        source_dir = tempfile.mkdtemp(dir=temp_dir)
        shutil.unpack_archive(archive_path, source_dir)
        source_dir = find_git_root(source_dir)
        binary_path = compile_bustools(
            source_dir, binary_path, cmake_arguments=cmake_arguments
        )
        results['bustools'] = binary_path
    return results
