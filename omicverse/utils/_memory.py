"""Shared memory-detection utilities for OmicVerse."""

from __future__ import annotations

import os

# Fraction of available CPU RAM that a single dense allocation may occupy.
# PCA and similar operations need ~2-3x the array size internally, so 0.3
# leaves headroom for temporaries.
AUTO_DENSE_CPU_MEM_FRACTION = 0.3

# Density threshold above which a sparse matrix should be converted to dense
# for better performance (sparse matrix ops have per-element overhead that
# only pays off when the matrix is actually sparse).
HIGH_DENSITY_SPARSE_THRESHOLD = 0.2

# Maximum element count for densification when memory detection fails.
# 500M float32 elements ≈ 2 GB — a safe ceiling for most machines.
MAX_SAFE_DENSE_ELEMENTS = 500_000_000


def get_available_memory() -> int | None:
    """Return available system memory in bytes, or ``None`` if unknown.

    Detection chain: psutil → /proc/meminfo → os.sysconf.
    Callers should treat ``None`` as "memory is unknown" and fall back to
    :data:`MAX_SAFE_DENSE_ELEMENTS` as a size-based guard.
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except (OSError, ValueError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, ValueError, OSError):
        pass
    return None
