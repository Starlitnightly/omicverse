r"""
Input/Output utilities for OmicVerse datasets.

Subpackages:
    general: Shared I/O helpers for tabular data and serialization.
    bulk: I/O helpers for bulk omics resources.
    single: I/O helpers for single-cell data.
    spatial: I/O helpers for spatial omics data.
"""

from . import bulk, general, single, spatial

__all__ = ["general", "bulk", "single", "spatial"]
