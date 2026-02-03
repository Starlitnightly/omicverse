r"""
Alignment analysis utilities.

This module provides comprehensive tools for fastq data processing and alignment including:
- Alignment with kb-python
- RNA velocity analysis with kb-python
- SRA download / conversion / QC / alignment / counting wrappers
"""

from .kb_api import single
from .prefetch import prefetch
from .fq_dump import fqdump
from .fastp import fastp
from .STAR import STAR
from .count import count

__all__ = [
    "single",
    "prefetch",
    "fqdump",
    "fastp",
    "STAR",
    "count",
]
