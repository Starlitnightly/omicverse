import functools
import os
import re
from typing import Callable

import scipy.io

from .config import get_bustools_binary_path, is_dry, is_validate
from .logging import logger
from .utils import run_executable

BUSTOOLS_INSPECT_PARSER = re.compile(r'^.*?(?P<count>[0-9]+)')


class ValidateError(Exception):
    pass


def validate_bus(path: str):
    """Verify if the provided BUS file is valid.

    A BUS file is considered valid when `bustools inspect` can read
    the file + it has > 0 BUS records.

    Args:
        path: Path to BUS file

    Raises:
        ValidateError: If the file failed verification
        subprocess.CalledProcessError: If the bustools command failed
    """
    command = [get_bustools_binary_path(), 'inspect', path]
    p, stdout, stderr = run_executable(command, quiet=True)
    match = BUSTOOLS_INSPECT_PARSER.match(stdout)
    if not match:
        raise ValidateError(
            f'bustools inspect output could not be parsed for {path}'
        )
    if int(match.groupdict().get('count', 0)) == 0:
        raise ValidateError(f'{path} has no BUS records')


def validate_mtx(path: str):
    """Verify if the provided Matrix Market (.mtx) file is valid.

    A BUS file is considered valid when the file can be read with `scipy.io.mmread`.

    Args:
        path: Path to mtx file

    Raises:
        ValidateError: If the file failed verification
    """
    try:
        scipy.io.mmread(path)
    except ValueError:
        raise ValidateError(f'{path} is not a valid matrix market file')


VALIDATORS = {
    '.bus': validate_bus,
    '.mtx': validate_mtx,
}


def validate(path: str):
    """Validate a file.

    This function is a wrapper around all validation functions.
    Given a path, it chooses the correct validation function.
    This function assumes the file exists.

    Args:
        path: Path to file

    Raises:
        ValidateError: If the file failed verification
    """
    # Validation is turned off.
    if not is_validate():
        return

    if not os.path.exists(path):
        raise ValidateError(f'{path} does not exist')

    ext = os.path.splitext(path)[1]
    if ext in VALIDATORS:
        VALIDATORS[ext](path)
        logger.debug(f'{path} passed validation')


def validate_files(pre: bool = True, post: bool = True) -> Callable:
    """Function decorator to validate input/output files.

    This function does not validate when the current run is a dry run.
    The decorated function is expected to return a dictionary of paths as
    values.

    Args:
        pre: Whether to validate input files, defaults to `True`
        post: Whether to validate output files, defaults to `True`

    Returns:
        Wrapped function
    """

    def wrapper(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):
            if not is_dry() and pre:
                for arg in list(args) + list(kwargs.values()):
                    if isinstance(arg, str) and os.path.exists(arg):
                        validate(arg)

            results = func(*args, **kwargs)

            # Assume results are in the form
            if not is_dry() and post:
                for path in results.values():
                    validate(path)

            return results

        return inner

    return wrapper
