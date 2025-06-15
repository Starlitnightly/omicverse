r"""
CyLib: High-performance computational library for omics data.

This module provides optimized computational utilities and fast implementations
of common algorithms used throughout the omicverse ecosystem. It includes both
Python and Cython-optimized functions for performance-critical operations.

Key features:
- Fast numerical computations for large-scale data
- Optimized algorithms for single-cell analysis
- Analytics and performance monitoring utilities
- Memory-efficient data structures
- GPU acceleration support where available

Main components:
    fast_utils: High-performance utility functions
    _analytics_sender: Analytics and telemetry utilities

Applications:
- Accelerated preprocessing of large datasets
- Fast neighborhood calculations
- Efficient similarity computations
- Performance monitoring and optimization
- Memory-efficient data transformations

Performance features:
- Cython-compiled critical functions
- Vectorized numpy operations
- Sparse matrix optimizations
- Memory mapping for large files
- Parallel processing support

Examples:
    >>> import omicverse as ov
    >>> # Fast utility functions are automatically available
    >>> # through other omicverse modules
    >>> 
    >>> # Direct usage (advanced)
    >>> from omicverse.cylib import fast_utils
    >>> result = fast_utils.fast_computation(data)

Notes:
    - Functions are primarily used internally by other modules
    - Provides significant speedups for large datasets
    - Maintains compatibility with standard numpy/scipy interfaces
    - Some functions may require compilation of Cython extensions
"""

from .fast_utils import *