r"""
VIA: Velocity-based cell fate determination and trajectory analysis.

This module implements VIA (Velocity Integration and Annotation), a method for
inferring cell fate decisions and developmental trajectories from single-cell 
RNA velocity data. VIA uses graph-based algorithms to model cell state transitions
and predict future cell states.

Key features:
- Single-cell RNA velocity analysis
- Trajectory inference and lineage tracing
- Cell fate probability calculation
- Terminal state prediction
- Pseudotime computation along trajectories
- Integration with scVelo and other velocity tools

Main components:
    core: Core VIA algorithms and trajectory inference
    examples: Example datasets and workflows
    utils_via: Utility functions for VIA analysis
    windmap: Wind map visualization of velocity fields
    plotting_via: Specialized plotting functions for trajectories
    datasets_via: Data loading and preprocessing utilities

Applications:
- Developmental biology trajectory analysis
- Cell fate decision characterization
- Pseudotime ordering along lineages
- Terminal state identification
- Velocity field visualization

Key algorithms:
- Graph-based trajectory inference
- Markov chain modeling of cell transitions
- Terminal state detection
- Pseudotime calculation
- Velocity field integration

Examples:
    >>> import omicverse as ov
    >>> import scanpy as sc
    >>> 
    >>> # Load data with velocity information
    >>> adata = sc.read_h5ad('data_with_velocity.h5ad')
    >>> 
    >>> # Initialize VIA
    >>> via = ov.via.core.VIA(
    ...     adata,
    ...     velocity_matrix=adata.layers['velocity']
    ... )
    >>> 
    >>> # Run trajectory analysis
    >>> via.run_via()
    >>> 
    >>> # Visualize results
    >>> ov.via.plotting_via.plot_trajectory(
    ...     adata,
    ...     color_by='via_pseudotime'
    ... )
    >>> 
    >>> # Generate wind map
    >>> ov.via.windmap.plot_windmap(adata)

Notes:
    - Requires velocity information (e.g., from scVelo)
    - Integrates with standard single-cell analysis workflows
    - Provides both local and global trajectory inference
    - Supports multiple trajectory starting points
"""

from . import core
from . import examples
from . import utils_via
from . import windmap
from . import plotting_via
from . import datasets_via

