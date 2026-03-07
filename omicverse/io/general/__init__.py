r"""General-purpose I/O helpers."""

from ._serialization import load, save
from ._tabular import read_csv

__all__ = ["read_csv", "save", "load"]

