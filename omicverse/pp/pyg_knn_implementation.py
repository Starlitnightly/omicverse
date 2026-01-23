#!/usr/bin/env python
"""
PyTorch Geometric KNN implementation for OmicVerse
Provides GPU-accelerated KNN search using PyG

Installation:
    pip install torch-geometric
    pip install torch-cluster  # Required for knn function

Alternative (manual KNN):
    If torch-cluster is not available, uses torch.cdist
"""

import torch
import numpy as np
from scipy import sparse


def pyg_knn_search(X, k=15, device='cuda', batch_size=None):
    """
    GPU-accelerated KNN search using PyTorch Geometric.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Data matrix of shape (n_samples, n_features)
    k : int
        Number of nearest neighbors
    device : str
        'cuda' or 'cpu'
    batch_size : int, optional
        Process data in batches to save memory

    Returns
    -------
    indices : np.ndarray
        KNN indices of shape (n_samples, k)
    distances : np.ndarray
        KNN distances of shape (n_samples, k)
    """
    # Convert to torch tensor
    if isinstance(X, np.ndarray):
        X_torch = torch.from_numpy(X).float()
    else:
        X_torch = X.float()

    # Move to device
    X_torch = X_torch.to(device)
    n_samples = X_torch.shape[0]

    # Try PyG knn first
    try:
        from torch_geometric.nn import knn as pyg_knn

        # Create batch indices (all samples in one batch)
        batch = torch.zeros(n_samples, dtype=torch.long, device=device)

        # Compute KNN
        edge_index = pyg_knn(X_torch, X_torch, k, batch, batch)

        # Convert to indices format
        indices = edge_index[1].view(n_samples, k)

        # Calculate distances
        row_idx = edge_index[0]
        col_idx = edge_index[1]
        x_rows = X_torch[row_idx]
        x_cols = X_torch[col_idx]
        edge_distances = torch.norm(x_rows - x_cols, dim=1)
        distances = edge_distances.view(n_samples, k)

        # Move to CPU and convert to numpy
        indices_np = indices.cpu().numpy()
        distances_np = distances.cpu().numpy()

        return indices_np, distances_np

    except ImportError:
        print("⚠️  torch-cluster not installed, using fallback torch.cdist")
        return torch_knn_fallback(X_torch, k, batch_size)


def torch_knn_fallback(X_torch, k=15, batch_size=None):
    """
    Fallback KNN implementation using torch.cdist.
    Works without torch-cluster but may use more memory.

    Parameters
    ----------
    X_torch : torch.Tensor
        Data on device, shape (n_samples, n_features)
    k : int
        Number of nearest neighbors
    batch_size : int, optional
        Process in batches to save memory

    Returns
    -------
    indices : np.ndarray
    distances : np.ndarray
    """
    n_samples = X_torch.shape[0]
    device = X_torch.device

    if batch_size is None or batch_size >= n_samples:
        # Compute all distances at once
        dist_matrix = torch.cdist(X_torch, X_torch)

        # Get top-k neighbors
        distances, indices = torch.topk(
            dist_matrix, k, dim=1, largest=False, sorted=True
        )

        return indices.cpu().numpy(), distances.cpu().numpy()

    else:
        # Process in batches to save memory
        all_indices = []
        all_distances = []

        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_X = X_torch[i:end_i]

            # Compute distances for this batch
            batch_dist = torch.cdist(batch_X, X_torch)

            # Get top-k
            batch_distances, batch_indices = torch.topk(
                batch_dist, k, dim=1, largest=False, sorted=True
            )

            all_indices.append(batch_indices.cpu())
            all_distances.append(batch_distances.cpu())

        indices = torch.cat(all_indices, dim=0).numpy()
        distances = torch.cat(all_distances, dim=0).numpy()

        return indices, distances


def torch_knn_transformer(n_neighbors=15, metric='euclidean', device='auto'):
    """
    sklearn-compatible transformer using PyG KNN.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors
    metric : str
        Distance metric (only 'euclidean' supported for now)
    device : str
        'auto', 'cuda', or 'cpu'

    Returns
    -------
    transformer : TorchKNNTransformer
        sklearn-compatible transformer
    """
    return TorchKNNTransformer(
        n_neighbors=n_neighbors,
        metric=metric,
        device=device
    )


class TorchKNNTransformer:
    """
    sklearn-compatible KNN transformer using PyTorch.

    Examples
    --------
    >>> transformer = TorchKNNTransformer(n_neighbors=15)
    >>> distances = transformer.fit_transform(X)
    >>> print(distances.shape)  # (n_samples, n_samples) sparse matrix
    """

    def __init__(self, n_neighbors=15, metric='euclidean', device='auto'):
        self.n_neighbors = n_neighbors
        self.metric = metric

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def fit(self, X, y=None):
        """Fit the model (no-op for KNN)."""
        return self

    def fit_transform(self, X, y=None):
        """
        Compute KNN and return sparse distance matrix.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features)

        Returns
        -------
        distances : scipy.sparse.csr_matrix
            Sparse distance matrix of shape (n_samples, n_samples)
        """
        # Compute KNN
        knn_indices, knn_distances = pyg_knn_search(
            X, k=self.n_neighbors, device=self.device
        )

        # Convert to sparse matrix
        n_samples = X.shape[0]

        # Create row indices (each row has n_neighbors entries)
        row_indices = np.repeat(np.arange(n_samples), self.n_neighbors)
        col_indices = knn_indices.ravel()
        distances = knn_distances.ravel()

        # Create sparse matrix
        distance_matrix = sparse.csr_matrix(
            (distances, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )

        return distance_matrix

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """
        Find k-neighbors (sklearn-compatible interface).

        Parameters
        ----------
        X : np.ndarray
            Query points
        n_neighbors : int, optional
            Number of neighbors to return
        return_distance : bool
            Whether to return distances

        Returns
        -------
        distances : np.ndarray (optional)
        indices : np.ndarray
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        indices, distances = pyg_knn_search(
            X, k=n_neighbors, device=self.device
        )

        if return_distance:
            return distances, indices
        else:
            return indices

    def get_params(self, deep=True):
        """Get parameters (sklearn-compatible)."""
        return {
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'device': self.device
        }

    def set_params(self, **params):
        """Set parameters (sklearn-compatible)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
