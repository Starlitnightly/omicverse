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
import importlib

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

_LAZY_ATTRS = {
    "BaseAlgorithm": ("._base_algorithm", "BaseAlgorithm"),
    "KNN_BBKNN": ("._bbknn", "KNN_BBKNN"),
    "CELLTYPIST": ("._celltypist", "CELLTYPIST"),
    "KNN_HARMONY": ("._harmony", "KNN_HARMONY"),
    "ONCLASS": ("._onclass", "ONCLASS"),
    "Random_Forest": ("._rf", "Random_Forest"),
    "KNN_SCANORAMA": ("._scanorama", "KNN_SCANORAMA"),
    "SCANVI_POPV": ("._scanvi", "SCANVI_POPV"),
    "KNN_SCVI": ("._scvi", "KNN_SCVI"),
    "Support_Vector": ("._svm", "Support_Vector"),
    "XGboost": ("._xgboost", "XGboost"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
