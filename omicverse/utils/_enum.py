'''
Copy from cellrank

'''
from typing import Any, Dict, Type, Tuple, Literal, Callable

from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from functools import wraps

_DEFAULT_BACKEND = "loky"
Backend_t = Literal["loky", "multiprocessing", "threading"]


class PrettyEnum(Enum):
    r"""Enum with a pretty __str__ and __repr__."""

    @property
    def v(self) -> Any:
        r"""Alias for value attribute."""
        return self.value

    def __repr__(self) -> str:
        return f"{self.value!r}"

    def __str__(self) -> str:
        return f"{self.value!s}"


def _pretty_raise_enum(cls: Type["ErrorFormatterABC"], func: Callable) -> Callable:
    r"""Wrap enum constructors to emit readable invalid-option errors.

    Parameters
    ----------
    cls : Type[ErrorFormatterABC]
        Enum class implementing custom error formatting.
    func : Callable
        Original ``__new__``/constructor function to wrap.

    Returns
    -------
    Callable
        Wrapped constructor that rewrites ``ValueError`` messages using
        ``cls._format``.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> "ErrorFormatterABC":
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            _cls, value, *_ = args
            e.args = (cls._format(value),)
            raise e

    if not issubclass(cls, ErrorFormatterABC):
        raise TypeError(f"Class `{cls}` must be subtype of `ErrorFormatterABC`.")
    elif not len(cls.__members__):
        # empty enum, for class hierarchy
        return func

    return wrapper


class ABCEnumMeta(EnumMeta, ABCMeta):
    r"""Metaclass for abstract enum classes."""
    def __call__(cls, *args, **kwargs):  # noqa
        if getattr(cls, "__error_format__", None) is None:
            raise TypeError(
                f"Can't instantiate class `{cls.__name__}` "
                f"without `__error_format__` class attribute."
            )
        return super().__call__(*args, **kwargs)

    def __new__(  # noqa: D102
        cls, clsname: str, superclasses: Tuple[type], attributedict: Dict[str, Any]
    ):
        res = super().__new__(cls, clsname, superclasses, attributedict)
        res.__new__ = _pretty_raise_enum(res, res.__new__)
        return res


class ErrorFormatterABC(ABC):
    r"""Abstract base class for error formatting in enums."""
    __error_format__ = "Invalid option `{!r}` for `{}`. Valid options are: `{}`."

    @classmethod
    def _format(cls, value) -> str:
        r"""Format an invalid-enum error message.

        Parameters
        ----------
        value : Any
            Invalid option value provided by the caller.

        Returns
        -------
        str
            Human-readable error message with allowed enum values.
        """
        return cls.__error_format__.format(
            value, cls.__name__, [m.value for m in cls.__members__.values()]
        )


class ModeEnum(str, ErrorFormatterABC, PrettyEnum, metaclass=ABCEnumMeta):
    r"""String-based enum with error formatting capabilities."""
    def _generate_next_value_(self, start, count, last_values):
        r"""Generate lowercase values for auto-enum members.

        Parameters
        ----------
        start : Any
            Initial value (unused).
        count : int
            Index of current enum value (unused).
        last_values : list
            Previously generated values (unused).

        Returns
        -------
        str
            Lowercase string of the current enum name.
        """
        return str(self).lower()
