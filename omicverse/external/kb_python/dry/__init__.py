import functools
from typing import Callable

from ..config import is_dry


def dryable(dry_func: Callable) -> Callable:
    """Function decorator to set a function as dryable.

    When this decorator is applied, the provided `dry_func` will be called
    instead of the actual function when the current run is a dry run.

    Args:
        dry_func: Function to call when it is a dry run

    Returns:
        Wrapped function
    """

    def wrapper(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):
            if not is_dry():
                return func(*args, **kwargs)
            else:
                return dry_func(*args, **kwargs)

        return inner

    return wrapper


def dummy_function(*args, **kwargs):
    """A dummy function that doesn't do anything and just returns.
    Used for making functions dryable.
    """
    return


def undryable_function(*args, **kwargs):
    """A dummy function that raises an exception. For use when a particular
    function is not dryable.

    Raises:
        Exception: Always
    """
    raise Exception('This function is not dryable.')
