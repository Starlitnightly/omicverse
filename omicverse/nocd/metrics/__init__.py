r"""
NoCD evaluation metrics for community detection.

This module provides comprehensive evaluation metrics for assessing the quality
of community detection results in single-cell data. It includes both supervised
metrics (when ground truth is available) and unsupervised metrics (intrinsic
quality measures).

Supervised metrics:
    - Accuracy-based measures for comparing with ground truth
    - Classification metrics (precision, recall, F1-score)
    - Information-theoretic measures (mutual information, entropy)
    - Clustering-specific metrics (ARI, silhouette score)

Unsupervised metrics:
    - Modularity and quality measures
    - Silhouette analysis
    - Calinski-Harabasz index
    - Davies-Bouldin index
    - Internal consistency measures

Features:
- Designed specifically for biological data characteristics
- Handles high-dimensional sparse data
- Supports both discrete and continuous evaluation
- Integrates with standard scikit-learn metrics
"""

from .supervised import *
from .unsupervised import *
