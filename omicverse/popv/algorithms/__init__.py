r"""
PopV algorithms collection for cell annotation and integration.

This module provides a comprehensive collection of algorithms for automated cell
type annotation and batch integration, implemented with a unified interface for
easy comparison and benchmarking.

Base classes:
    BaseAlgorithm: Abstract base class for all PopV algorithms

K-nearest neighbor methods:
    KNN_BBKNN: Batch-balanced k-nearest neighbors integration
    KNN_HARMONY: Harmony-based batch correction with kNN
    KNN_SCANORAMA: Scanorama integration with kNN classification
    KNN_SCVI: scVI latent space with kNN classification

Machine learning classifiers:
    CELLTYPIST: Automated cell type classification
    Random_Forest: Random forest classifier for cell types
    Support_Vector: Support vector machine classification
    XGboost: Gradient boosting classifier

Deep learning methods:
    SCANVI_POPV: Semi-supervised variational inference
    ONCLASS: Ontology-guided cell type annotation

Features:
- Unified interface for all algorithms
- Automatic hyperparameter optimization
- Cross-validation and benchmarking support
- Integration with popular single-cell frameworks
- Comprehensive evaluation metrics
"""

from ._base_algorithm import BaseAlgorithm
from ._bbknn import KNN_BBKNN
from ._celltypist import CELLTYPIST
from ._harmony import KNN_HARMONY
from ._onclass import ONCLASS
from ._rf import Random_Forest
from ._scanorama import KNN_SCANORAMA
from ._scanvi import SCANVI_POPV
from ._scvi import KNN_SCVI
from ._svm import Support_Vector
from ._xgboost import XGboost

__all__ = [
    "BaseAlgorithm",
    "CELLTYPIST",
    "KNN_BBKNN",
    "KNN_HARMONY",
    "KNN_SCANORAMA",
    "KNN_SCVI",
    "ONCLASS",
    "Random_Forest",
    "SCANVI_POPV",
    "Support_Vector",
    "XGboost",
]
