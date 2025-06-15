r"""
NoCD: Non-overlapping Community Detection for single-cell data.

This module implements non-overlapping community detection algorithms specifically
designed for single-cell omics data. It provides neural network-based approaches
for identifying discrete cell populations and communities in high-dimensional
biological data.

Key features:
- Neural network-based community detection
- Non-overlapping cluster identification
- Specialized metrics for biological data evaluation
- Scalable training procedures for large datasets
- Integration with single-cell analysis workflows

Main components:
    data: Data loading and preprocessing utilities
    nn: Neural network architectures for community detection
    metrics: Evaluation metrics for clustering quality
    sampler: Data sampling strategies for training
    train: Training procedures and optimization
    utils: Utility functions and helper methods

Applications:
- Cell type identification in single-cell RNA-seq
- Population structure analysis
- Discrete state detection in developmental trajectories
- Batch effect-robust clustering

Examples:
    >>> import omicverse as ov
    >>> # Load and preprocess data
    >>> loader = ov.nocd.data.load_data(adata)
    >>> 
    >>> # Initialize neural network model
    >>> model = ov.nocd.nn.CommunityDetectionNet(
    ...     input_dim=adata.n_vars,
    ...     hidden_dim=64,
    ...     n_communities=10
    ... )
    >>> 
    >>> # Train the model
    >>> trainer = ov.nocd.train.Trainer(model, loader)
    >>> trainer.fit()
    >>> 
    >>> # Evaluate results
    >>> metrics = ov.nocd.metrics.evaluate_clustering(
    ...     predictions, ground_truth
    ... )

Notes:
    - Designed for discrete, non-overlapping community structure
    - Optimized for high-dimensional biological data
    - Provides both supervised and unsupervised training modes
    - Supports GPU acceleration for large-scale analysis
"""

from . import data
from . import nn
from . import metrics
from . import sampler
from . import train
from . import utils
